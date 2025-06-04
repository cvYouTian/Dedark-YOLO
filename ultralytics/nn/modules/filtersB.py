import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from .util_filters import rgb2lum, tanh_range, lerp
import cv2
import math


__all__ = ("Filter", "DeDarkFilter", "UsmFilter", "GammaFilter", "ImprovedWhiteBalanceFilter", "ContrastFilter", "ToneFilter")

class Filter(nn.Module):
    def __init__(self,  cfg):
        super().__init__()
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

        return features[:, self.get_begin_filter_parameter():(
                    self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
            features[:,
            self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

    def filter_param_regressor(self, features):
        assert False

    def process(self, img, param, defog, IcA):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking
    def forward(self,img,
              img_features=None,
              defog_A=None,
              IcA=None,
              specified_parameter=None,
              high_res=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            assert not self.use_masking()
            filter_parameters = specified_parameter
            mask_parameters = torch.zeros(
                (1, self.get_num_mask_parameters()), dtype=torch.float32)
        if high_res is not None:
            pass
        debug_info = {}
        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters[0]


        low_res_output = self.process(img, filter_parameters, defog_A, IcA)
        if high_res is not None:
            if self.no_high_res():
                high_res_output = high_res
            else:
                self.high_res_mask = self.get_mask(high_res, mask_parameters)
                # high_res_output = lerp(high_res,
                #                        self.process(high_res, filter_parameters, defog, IcA),
                #                        self.high_res_mask)
        else:
            high_res_output = None
        # return low_res_output, high_res_output, debug_info
        return low_res_output, filter_parameters

    def use_masking(self):
        return self.cfg.masking

    def get_num_mask_parameters(self):
        return 6

    # Input: no need for tanh or sigmoid
    # Closer to 1 values are applied by filter more strongly
    # no additional TF variables inside
    def get_mask(self, img, mask_parameters):
        if not self.use_masking():
            print('* Masking Disabled')
            return torch.ones((1, 1, 1, 1), dtype=torch.float32)
        else:
            print('* Masking Enabled')
        with self.name_scope('mask'):
            # Six parameters for one filter
            filter_input_range = 5
            assert mask_parameters.shape[1] == self.get_num_mask_parameters()
            mask_parameters = tanh_range(
                l=-filter_input_range, r=filter_input_range,
                initial=0)(mask_parameters)
            size = list(map(int, img.shape[1:3]))
            grid = torch.zeros([1] + size + [2], dtype=torch.float32)

            shorter_edge = min(size[0], size[1])
            for i in range(size[0]):
                for j in range(size[1]):
                    grid[0, i, j, 0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                    grid[0, i, j, 1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
            grid = torch.constant(grid)
            inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
                  grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
                  mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
                  mask_parameters[:, None, None, 3, None] * 2
            # Sharpness and inversion
            inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4, None] / filter_input_range
            mask = torch.sigmoid(inp)
            # Strength
            mask = mask * (
                    mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
                    0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
            print('mask', mask.shape)
        return mask

    def visualize_mask(self, debug_info, res):
        return cv2.resize(
            debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
            dsize=res,
            interpolation=cv2.cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(
            canvas,
            text, (30, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 0),
            thickness=5)
        return canvas


class UsmFilter(Filter):
    def __init__(self, net, cfg):
        super().__init__(cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param, defog_A, IcA):
        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.cast(-radius, radius + 1, dtype=dtype)
            k = torch.exp(-0.5 * torch.square(x / sigma))
            k = k / torch.reduce_sum(k)
            return torch.expand_dims(k, 1) * k

        kernel_i = make_gaussian_2d_kernel(5)
        # print('kernel_i.shape', kernel_i.shape)
        kernel_i = kernel_i.unsqueeze(0).unsqueeze(1).to(img.device)

        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        outputs = []
        for channel_idx in range(3):
            data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
            data_c = F.conv2d(data_c, kernel_i, stride=1)
            outputs.append(data_c)
        output = torch.cat(outputs, dim=1)
        img_out = (img - output) * param[:, :, None, None] + img
        return img_out



class DedarkFilter(Filter):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'DF'
        self.begin_filter_parameter = cfg.defog_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.defog_range)(features)

    def process(self, img, param, defog_A, IcA):
        print('      defog_A:', img.shape)
        print('      defog_A:', IcA.shape)
        print('      defog_A:', defog_A.shape)

        tx = 1 - param[:, :, None, None] * IcA
        # tx = 1 - 0.5*IcA

        tx_1 = torch.tile(tx, [1, 1, 1, 3])
        return (img - defog_A[:, :, None, None]) / torch.clamp(tx_1, min=0.01) + defog_A[:, :, None, None,]


class GammaFilter(Filter):  # gamma_param is in [-gamma_range, gamma_range]

    def __init__(self, net, cfg):
        super().__init__(cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.cfg.gamma_range)
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param, defog_A, IcA):
        param_1 = param.repeat(1, 3)
        return torch.pow(torch.clamp(img, 0.0001), param_1[:, :, None, None])
        # return img

class ImprovedWhiteBalanceFilter(Filter):

    def __init__(self, net, cfg):
        super().__init__(cfg)
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
        # mask = np.array(((1, 0, 1)), dtype=np.float32).reshape(1, 3)
        print(mask.shape)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling *= 1.0 / (
                                       1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                       0.06 * color_scaling[:, 2])[:, None]
        return color_scaling

    def process(self, img, param, defog, IcA):
        return img * param[:, :, None, None]


class ToneFilter(Filter):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param

        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = features.view(-1, 1, self.cfg.curve_steps).unsqueeze(1).unsqueeze(2)
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param, defog, IcA):
        # img = tf.minimum(img, 1.0)
        tone_curve = param
        tone_curve_sum = torch.sum(tone_curve, dim=4) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                           * param[:, :, :, :, i]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        img = total_image
        return img


class ContrastFilter(Filter):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return torch.tanh(features)

    def process(self, img, param, defog, IcA):
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)
        # luminance = torch.minimum(torch.maximum(rgb2lum(img), 0.0), 1.0)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])

