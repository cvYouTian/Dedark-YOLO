import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import cv2
import os
from .config_lowlight import cfg
from .common import ExtractParameters2

__all__ = ("lowlight_recovery")


class lowlight_recovery(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, lowlight_param=1.0):
        super().__init__()
        self.extractor = ExtractParameters2(cfg)
        self.filters = nn.ModuleList(cfg.filters)
        self.lowlight_param = random.uniform(5, 10)
        self.save_dir = "filtered_images"  # 保存目录
        os.makedirs(self.save_dir, exist_ok=True)  # 创建目录

    def save_image(self, filtered_image, batch_idx):
        """保存 filtered_image 的每张图像到磁盘"""
        filtered_image = filtered_image.detach().cpu().numpy()  # [batch_size, 3, h, w]
        batch_size = filtered_image.shape[0]
        for i in range(batch_size):
            img = filtered_image[i].transpose(1, 2, 0)  # [h, w, 3]
            img = (img * 255).astype(np.uint8)  # 转换为 [0, 255]
            img = img[:, :, ::-1]  # RGB to BGR for OpenCV
            save_path = os.path.join(self.save_dir, f"filtered_{batch_idx}_{i}.jpg")
            cv2.imwrite(save_path, img)
            print(f"Saved image: {save_path}")

    def forward(self, x, batch_idx=0):
        # Ensure input is on the correct device
        self.to(x.device)

        # Apply lowlight transformation
        input_data = torch.pow(x, self.lowlight_param)

        # Interpolate to fixed size
        input_data_resized = F.interpolate(input_data, size=(256, 256), mode='bilinear', align_corners=False)

        # Feature extraction
        filter_features = self.extractor(input_data_resized)

        # Apply filters
        filtered_image = input_data.clone()
        for filter in self.filters:
            filtered_image, _ = filter(filtered_image, filter_features)

        # Ensure output in [0, 1]
        filtered_image = torch.clamp(filtered_image, 0, 1)

        # Save filtered images
        # with torch.no_grad():
        #     self.save_image(filtered_image, batch_idx)

        # Compute recovery loss
        recovery_loss = F.mse_loss(filtered_image, x)  # 使用输入作为目标

        return filtered_image, recovery_loss