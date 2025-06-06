from copy import copy
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
import cv2
import math


class DetectionTrainer(BaseTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    # def preprocess_batch(self, batch):
    #     """Preprocesses a batch of images by scaling and converting to float."""
    #     batch['clean_img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
    #     batch["img"] = torch.pow(batch["clean_img"], self.dark_param)
    #
    #     # recover_loss = torch.sum((batch["img"] - batch["clean_img"]) ** 2)
    #     recover_loss = F.mse_loss(batch["img"], batch["clean_img"])
    #     batch["recovery_loss_batch"] = recover_loss
    #
    #     return batch
    #

    def DarkChannel(self, im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        return dc

    def AtmLight(self, im, dark):
        [h, w] = im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort(0)
        indices = indices[(imsz - numpx):imsz]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def DarkIcA(self, im, A):
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[ind, :, :] = im[ind, :, :] / A[0, ind]
        return self.DarkChannel(im3)

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['clean_img'] = batch['img'].to(self.device, non_blocking=True).float() / 255

        if hasattr(self.args, 'dedark_FLAG') and self.args.dedark_FLAG and self.args.lowlight_FLAG:
            batch_size = batch['clean_img'].shape[0]
            height = batch['clean_img'].shape[2]
            width = batch['clean_img'].shape[3]

            batch["clean_img"] = torch.pow(batch["clean_img"], self.args.dark_param)

            clean_imgs_np = (batch['clean_img'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # Initialize arrays like in the first file
            defog_A = np.zeros((batch_size, 3))
            IcA = np.zeros((batch_size, height, width))

            # Process each image in batch
            for i in range(batch_size):
                dark_i = self.DarkChannel(clean_imgs_np [i])
                defog_A_i = self.AtmLight(clean_imgs_np[i], dark_i)
                IcA_i = self.DarkIcA(clean_imgs_np[i], defog_A_i)
                defog_A[i, ...] = defog_A_i
                IcA[i, ...] = IcA_i

            # Convert back to PyTorch tensors
            batch['dedark_A'] = torch.from_numpy(defog_A).float().to(self.device)
            batch['IcA'] = torch.from_numpy(np.expand_dims(IcA, axis=-1)).permute(0, 3, 1, 2).float().to(self.device)

            # Use the processed image (same as clean_img in this case, following the original logic)
            batch["img"] = batch["clean_img"]

        elif hasattr(self.args, "lowlight_FLAG") and self.args.lowlight_FLAG:
            batch["img"] = torch.pow(batch["clean_img"], self.args.dark_param)

        else:
            batch["img"] = batch["clean_img"]

        recover_loss = F.mse_loss(batch["img"], batch["clean_img"])
        batch["recovery_loss_batch"] = recover_loss

        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        # model = LowLightDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()