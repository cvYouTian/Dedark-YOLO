import torch
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_file.Utils.utils as utils


class YoloDataset(Dataset):
    def __init__(self, dataset_type, cfg):
        self.dataset_type = dataset_type
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 100

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.train_input_size = self.input_sizes
        self.train_output_sizes = self.train_input_size // self.strides

    def __len__(self):
        return self.num_samples

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        return annotations


    def __getitem__(self, idx):

        annotation = self.annotations[idx]
        image, bboxes = self.parse_annotation(annotation)
        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
        # 这里是第一次调整为pytorch的尺寸
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()

        return image_tensor, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        image = np.array(cv2.imread(image_path))
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size] * 2, np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area


    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float32)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes






















#     def bbox_iou(self, boxes1, boxes2):
#         boxes1_area = boxes1[..., 2] * boxes1[..., 3]
#         boxes2_area = boxes2[..., 2] * boxes2[..., 3]
#
#         boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
#                                  boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
#         boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
#                                  boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)
#
#         left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
#         right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
#
#         inter_section = np.maximum(right_down - left_up, 0.0)
#         inter_area = inter_section[..., 0] * inter_section[..., 1]
#         union_area = boxes1_area + boxes2_area - inter_area
#
#         return inter_area / union_area
#
#     def preprocess_true_boxes(self, bboxes):
#         label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
#                            5 + self.num_classes)) for i in range(3)]
#         bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
#         bbox_count = np.zeros((3,))
#
#         for bbox in bboxes:
#             bbox_coor = bbox[:4]
#             bbox_class_ind = int(bbox[4])
#
#             onehot = np.zeros(self.num_classes, dtype=np.float32)
#             onehot[bbox_class_ind] = 1.0
#             uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
#             deta = 0.01
#             smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
#
#             bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5,
#                                         bbox_coor[2:] - bbox_coor[:2]], axis=-1)
#             bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]
#
#             iou = []
#             exist_positive = False
#             for i in range(3):
#                 anchors_xywh = np.zeros((self.anchor_per_scale, 4))
#                 anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(int) + 0.5
#                 anchors_xywh[:, 2:4] = self.anchors[i]
#
#                 iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
#                 iou.append(iou_scale)
#                 iou_mask = iou_scale > 0.3
#
#                 if np.any(iou_mask):
#                     xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(int)
#
#                     label[i][yind, xind, iou_mask, :] = 0
#                     label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
#                     label[i][yind, xind, iou_mask, 4:5] = 1.0
#                     label[i][yind, xind, iou_mask, 5:] = smooth_onehot
#
#                     bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
#                     bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
#                     bbox_count[i] += 1
#
#                     exist_positive = True
#
#             if not exist_positive:
#                 best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
#                 best_detect = int(best_anchor_ind / self.anchor_per_scale)
#                 best_anchor = int(best_anchor_ind % self.anchor_per_scale)
#                 xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(int)
#
#                 label[best_detect][yind, xind, best_anchor, :] = 0
#                 label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
#                 label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
#                 label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot
#
#                 bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
#                 bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
#                 bbox_count[best_detect] += 1
#
#         label_sbbox, label_mbbox, label_lbbox = label
#         sbboxes, mbboxes, lbboxes = bboxes_xywh
#         return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
#
#
# def create_dataloader(dataset_type, batch_size=None):
#     dataset = YoloDataset(dataset_type)
#     batch_size = batch_size or dataset.batch_size
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True if dataset_type == 'train' else False,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=lambda x: tuple(zip(*x))  # Handle batches
#     )
#     return dataloader