import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from .config_lowlight import cfg
from .common import ExtractParameters2

device = "cuda" if torch.cuda.is_available() else "cpu"

class lowlight_recovery(nn.Module):
    def __init__(self, input=3, output=3, lowlight_param=1):
        super().__init__()
        # 确保extractor在GPU上
        self.extractor = ExtractParameters2(cfg)

        # 确保所有filter都在GPU上
        self.filters = nn.ModuleList([f for f in cfg.filters])

        self.lowlight_param = lowlight_param if lowlight_param == 1 else random.uniform(1.5, 5)

        # 确保所有参数在GPU上
        # self.to(device)  # 这是关键修复
    def forward(self, x):
        if x.device == "cuda":

        else:
            # 确保输入在模型设备上
            # x = x.to(device)
            input_data_clean = x  # 保留原始数据

            # 使用PyTorch操作替代NumPy（保持自动微分）
            input_data = torch.pow(x, self.lowlight_param)
            filtered_image_batch = input_data.clone()

            # 插值操作
            input_data = F.interpolate(input_data, size=(256, 256), mode='bilinear')

            # 特征提取
            filter_features = self.extractor(input_data)

            # 应用过滤器（确保所有输入输出在相同设备）
            filter_parameters = []
            for filter in self.filters:
                filtered_image_batch, param = filter(filtered_image_batch, filter_features)
                filter_parameters.append(param)

            # 计算损失（显式设备同步）
            recovery_loss = F.mse_loss(filtered_image_batch, input_data_clean)

        return filtered_image_batch