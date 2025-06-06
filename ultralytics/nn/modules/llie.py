import torch.nn as nn
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
            filtered_image, _ = filter(filtered_image, filter_features, dedark_A, IcA)

        return filtered_image