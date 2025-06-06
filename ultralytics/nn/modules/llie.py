import torch.nn as nn
import torch
import torch.nn.functional as F
from .filter_cfg import cfg
from .common import ExtractParameters2


__all__ = ("lowlight_recovery")


class lowlight_recovery(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.extractor = ExtractParameters2(cfg)
        self.filters = nn.ModuleList(cfg.filters)

    def forward(self, x, dedark_A=None, IcA=None):
        """
        修改forward方法以接受额外的去雾参数，并处理None值情况

        Args:
            x: 输入图像 [B, C, H, W]
            dedark_A: 大气光强度 [B, 3] 或 None
            IcA: 暗通道相关信息 [B, 1, H, W] 或 None
        """
        # Ensure input is on the correct device

        self.to(x.device)
        input_data = x.clone()
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        if dedark_A is None:
            # 创建默认的大气光值 [B, 3]
            dedark_A = torch.ones(batch_size, 3, device=x.device) * 0.8

        if IcA is None:
            # 创建默认的暗通道值 [B, 1, H, W]
            IcA = torch.ones(batch_size, 1, height, width, device=x.device) * 0.5

        # Interpolate to fixed size
        input_data_resized = F.interpolate(input_data, size=(256, 256), mode='bilinear', align_corners=False)

        # Feature extraction
        filter_features = self.extractor(input_data_resized)

        # Apply filters
        filtered_image = input_data.clone()

        for filter in self.filters:
            filtered_image, _ = filter(filtered_image, filter_features, dedark_A, IcA)

        return filtered_image