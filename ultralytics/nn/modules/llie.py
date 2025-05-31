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
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.extractor = ExtractParameters2(cfg)
        self.filters = nn.ModuleList(cfg.filters)
        self.save_dir = "filtered_images"  # 保存目录
        os.makedirs(self.save_dir, exist_ok=True)  # 创建目录

    def forward(self, x):
        # Ensure input is on the correct device
        self.to(x.device)
        input_data = x.clone()

        # Interpolate to fixed size
        input_data_resized = F.interpolate(input_data, size=(256, 256), mode='bilinear', align_corners=False)

        # Feature extraction
        filter_features = self.extractor(input_data_resized)

        # Apply filters
        filtered_image = input_data.clone()
        for filter in self.filters:
            filtered_image, _ = filter(filtered_image, filter_features)

        return filtered_image