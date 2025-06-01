# Ultralytics YOLO 🚀, AGPL-3.0 license
import torch
import numpy as np
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ROOT, ops
from pathlib import Path
import cv2


class DetectionPredictor(BasePredictor):
    # 后处理使用非极大值抑制
    def postprocess(self, preds, img, orig_imgs, filtered_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []

        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            # filtered_img = filtered_imgs[i] if isinstance(filtered_imgs, list) else filtered_imgs

            filtered_img = np.clip(filtered_imgs[0].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255)



            filtered_img = filtered_img.astype(np.uint8)[..., ::-1]
            filtered_img = np.ascontiguousarray(filtered_img)

            # 定义保存路径
            # save_dir = Path('runs/detect/exp/filtered')
            # save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
            # save_path = save_dir / f'filtered_img{i}.jpg'
            #
            # # 保存图像
            # cv2.imwrite(str(save_path), filtered_img)
            # print(f"Saved filtered image to {save_path}")

            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], filtered_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=filtered_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or '/home/youtian/Documents/pro/pyCode/ultralytics-YOLOv8/runs/detect/train6/weights/best.pt'
    source = "/home/youtian/Documents/pro/pyCode/datasets/darkpic/192.168.39.20_20240726_195724_1337826_2.jpg"
    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict(use_python=True)
