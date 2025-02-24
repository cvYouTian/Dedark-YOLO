import torch
import torch.nn.functional as F
from pytorch_file.Utils.util_filters import lrelu, rgb2lum, tanh_range, lerp
import math


# ---------------------- 基类 Filter ----------------------
class Filter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_filter_parameters = None
        self.short_name = None
        self.begin_filter_parameter = 0

    def get_short_name(self):
        return self.short_name

    def get_num_filter_parameters(self):
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def filter_param_regressor(self, features):
        raise NotImplementedError

    def process(self, img, param):
        raise NotImplementedError

    def apply(self, img, img_features=None, specified_parameter=None):
        if img_features is not None:
            filter_params = self.filter_param_regressor(img_features)
        else:
            filter_params = specified_parameter
        return self.process(img, filter_params)

    def get_mask(self, img, mask_params):
        """生成空间掩码 (PyTorch 优化版)"""
        if not self.cfg.masking:
            return torch.ones_like(img[:, :1, :, :])

        # 生成网格坐标
        B, C, H, W = img.shape
        y_coords = torch.linspace(-1, 1, H, device=img.device)
        x_coords = torch.linspace(-1, 1, W, device=img.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)  # [1,H,W,2]

        # 计算掩码参数
        lum = rgb2lum(img).unsqueeze(1)  # [B,1,H,W]
        inp = (grid[..., 0] * mask_params[:, 0].view(-1, 1, 1, 1) +
               grid[..., 1] * mask_params[:, 1].view(-1, 1, 1, 1) +
               (lum - 0.5) * mask_params[:, 2].view(-1, 1, 1, 1) +
               mask_params[:, 3].view(-1, 1, 1, 1) * 2)

        inp *= self.cfg.maximum_sharpness * mask_params[:, 4].view(-1, 1, 1, 1)
        mask = torch.sigmoid(inp)
        mask = mask * (mask_params[:, 5].view(-1, 1, 1, 1) * 0.5 + 0.5) * (
                    1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
        return mask


# ---------------------- 具体滤镜实现 ----------------------
class UsmFilter(Filter):
    """非锐化掩模 (PyTorch 优化实现)"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

        # 预计算高斯核 (固定参数)
        self.gauss_kernel = self._make_gaussian_kernel(5).to(cfg.device)

    def _make_gaussian_kernel(self, sigma):
        """创建2D高斯核 (支持梯度计算)"""
        radius = 12
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        return (k.unsqueeze(1) * k.unsqueeze(0)).view(1, 1, 25, 25)

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param):
        # 维度调整: PyTorch 使用 NCHW 格式
        B, C, H, W = img.shape

        # 反射填充
        padded = F.pad(img, (12, 12, 12, 12), mode='reflect')

        # 分离通道处理
        blurred = F.conv2d(padded.view(B * C, 1, H + 24, W + 24),
                           self.gauss_kernel,
                           padding=0).view(B, C, H, W)

        # 锐化处理
        return img + (img - blurred) * param.view(-1, 1, 1, 1)


class GammaFilter(Filter):
    """伽马校正"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_range = math.log(self.cfg.gamma_range)
        return torch.exp(tanh_range(-log_range, log_range)(features))

    def process(self, img, param):
        return torch.clamp(img, 1e-5) ** param.view(-1, 1, 1, 1)


class ImprovedWhiteBalanceFilter(Filter):
    """改进的白平衡"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'W'
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = 3
        self.channel_mask = torch.tensor([[0.0, 1.0, 1.0]], device=cfg.device)

    def filter_param_regressor(self, features):
        log_range = 0.5
        scaled = tanh_range(-log_range, log_range)(features * self.channel_mask)
        color_scaling = torch.exp(scaled)

        # 亮度归一化
        lum = 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2]
        return color_scaling / (lum.view(-1, 1) + 1e-5)

    def process(self, img, param):
        return img * param.view(-1, 3, 1, 1)


class ToneFilter(Filter):
    """色调曲线调整"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param
        self.num_filter_parameters = cfg.curve_steps
        self.step_size = 1.0 / cfg.curve_steps

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.tone_curve_range)(features.view(-1, self.num_filter_parameters))

    def process(self, img, param):
        B, C, H, W = img.shape
        param = param.view(B, 1, 1, self.num_filter_parameters)

        # 分区间处理
        total = torch.zeros_like(img)
        for i in range(self.cfg.curve_steps):
            lower = i * self.step_size
            upper = (i + 1) * self.step_size
            mask = (img >= lower) & (img < upper)
            total += mask * (img - lower) * param[:, :, :, :, i]

            # 归一化
        return total * self.cfg.curve_steps / (param.sum(dim=-1, keepdim=True) + 1e-5)


class ContrastFilter(Filter):
    """对比度调整"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return torch.tanh(features) * 0.5 + 0.5  # 限制到[0,1]

    def process(self, img, param):
        lum = rgb2lum(img).unsqueeze(1)
        contrast_lum = (-torch.cos(math.pi * lum) + 1) * 0.5
        return lerp(img, img / (lum + 1e-5) * contrast_lum, param.view(-1, 1, 1, 1))