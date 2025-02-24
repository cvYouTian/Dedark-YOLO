import torch
import torch.nn.functional as F
import math

def tanh_range(l, r, initial=None):
    def forward(x):
        return torch.tanh(x) * (r - l) / 2 + (r + l) / 2
    return forward

def rgb2lum(img):
    return 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :, :] + 0.06 * img[:, 2, :, :]

def lerp(a, b, l):
    return (1 - l) * a + l * b

class Filter:
    def __init__(self, net, cfg):
        self.cfg = cfg
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def extract_parameters(self, features):
        return (features[:, self.get_begin_filter_parameter():(
                    self.get_begin_filter_parameter() + self.get_num_filter_parameters())],
                features[:, self.get_begin_filter_parameter():(
                            self.get_begin_filter_parameter() + self.get_num_filter_parameters())])

    def filter_param_regressor(self, features):
        raise NotImplementedError

    def process(self, img, param):
        raise NotImplementedError

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    def apply(self, img, img_features=None, specified_parameter=None, high_res=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            assert not self.use_masking()
            filter_parameters = specified_parameter
            mask_parameters = torch.zeros((1, self.get_num_mask_parameters()), dtype=torch.float32)

        if high_res is not None:
            pass  # 高分辨率处理逻辑

        debug_info = {}
        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters

        low_res_output = self.process(img, filter_parameters)

        if high_res is not None:
            if self.no_high_res():
                high_res_output = high_res
            else:
                self.high_res_mask = self.get_mask(high_res, mask_parameters)
                high_res_output = lerp(high_res, self.process(high_res, filter_parameters), self.high_res_mask)
        else:
            high_res_output = None

        return low_res_output, filter_parameters

    def use_masking(self):
        return self.cfg.masking

    def get_num_mask_parameters(self):
        return 6

    def get_mask(self, img, mask_parameters):
        if not self.use_masking():
            print('* Masking Disabled')
            return torch.ones((1, 1, 1, 1), dtype=torch.float32)
        else:
            print('* Masking Enabled')

        filter_input_range = 5
        assert mask_parameters.shape == self.get_num_mask_parameters()
        mask_parameters = tanh_range(l=-filter_input_range, r=filter_input_range)(mask_parameters)
        size = list(img.shape[1:3])
        grid = torch.zeros((1, size, size, 2), dtype=torch.float32)

        shorter_edge = min(size, size)
        for i in range(size):
            for j in range(size):
                grid[0, i, j, 0] = (i + (shorter_edge - size) / 2.0) / shorter_edge - 0.5
                grid[0, i, j, 1] = (j + (shorter_edge - size) / 2.0) / shorter_edge - 0.5

        grid = grid.to(img.device)
        inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
              grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
              mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
              mask_parameters[:, None, None, 3, None] * 2

        inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4, None] / filter_input_range
        mask = torch.sigmoid(inp)
        mask = mask * (mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 + 0.5) * (
                    1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
        return mask


class UsmFilter(Filter):
    def __init__(self, net, cfg):
        super().__init__(net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param):
        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=dtype)
            k = torch.exp(-0.5 * torch.square(x / sigma))
            k = k / torch.sum(k)
            return torch.outer(k, k)

        kernel_i = make_gaussian_2d_kernel(5)
        kernel_i = kernel_i.unsqueeze(0).unsqueeze(0)
        kernel_i = kernel_i.expand(1, 1, -1, -1)

        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        outputs = []
        for channel_idx in range(3):
            data_c = padded[:, channel_idx:channel_idx+1, :, :]
            data_c = F.conv2d(data_c, kernel_i, stride=1, padding=0)
            outputs.append(data_c)

        output = torch.cat(outputs, dim=1)
        img_out = (img - output) * param[:, None, None, :] + img
        return img_out