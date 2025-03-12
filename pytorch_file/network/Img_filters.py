import torch
import torch.nn.functional as F
from pyexpat import features
from torch import nn
from pytorch_file.Utils.util_filters import lrelu, rgb2lum, tanh_range, lerp
import math
import numpy as np


class Filters(nn.Module):
    """
    net:input_processed
    config:cfg

    return:
    """
    def __init__(self, net, config):
        super(Filters, self).__init__()
        self.cfg = config
        self.num_filter_parameters = None
        self.short_name = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name , "short_name has to be set"
        return self.short_name

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def get_num_filter_parameters(self):
        assert self.num_filter_parameter is not None, "num_filter_parameter is None"
        return self.num_filter_parameter

    def extract_parameters(self, features):
        # 抽取滤波参数的特征
        # (feature, feature)
        return (features[:, self.get_begin_filter_parameter():(
                self.get_begin_filter_parameter() + self.get_num_filter_parameters())],
                features[:,self.get_begin_filter_parameter():
                           (self.get_begin_filter_parameter() + self.get_num_filter_parameters())])

    def filter_param_regressor(self, features):
        assert False

    def process(self, img, param):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    def use_masking(self):
        return self.cfg.masking

    def get_num_mask_parameters(self):
        return 6

    def get_mask(self, img, mask_parameters):
        #
        if not self.use_masking():
            print('* Masking Disabled')
            return torch.ones((1, 1, 1, 1), dtype=torch.float32)
        else:
            print('* Masking Enabled')

        # Six parameters for one filter
        filter_input_range = 5
        assert mask_parameters.shape[1] == self.get_num_mask_parameters()
        # 使用的了一个新的激活函数来处理mask_parameters
        mask_parameters = tanh_range(
            l=-filter_input_range, r=filter_input_range,
            initial=0)(mask_parameters)
        # TODO:这里要检查是不是pytorch的矩阵形式
        size = list(map(int, img.shape[1:3]))
        grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

        shorter_edge = min(size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                grid[0, i, j,
                0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                grid[0, i, j,
                1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
        grid = torch.tensor(grid, dtype=torch.float32)
        # Ax + By + C * L + D
        inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
              grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
              mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
              mask_parameters[:, None, None, 3, None] * 2
        # Sharpness and inversion
        inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4,
                                            None] / filter_input_range
        mask = torch.sigmoid(inp)
        # Strength
        mask = mask * (
                mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
                0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
        print('mask', mask.shape)

        return mask

    def forward(self, img, img_features = None, specified_parameter = None, high_res = None):
        assert (img_features is None) ^ (specified_parameter is None), "Error"

        if img_features is not None:
            # [:, N], [:, N]
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)

        else:
            assert not self.use_masking()
            filter_parameters = specified_parameter
            # 做了一个空的mask参数矩阵，全零【1, N】
            mask_parameters = torch.zeros((1, self.get_num_filter_parameters()), dtype=torch.float32)

        debug_info = dict()
        if self.debug_info_batched():
            debug_info["filter_parameters"] = filter_parameters
        else:
            debug_info["filter_parameters"] = filter_parameters[0]

        # output by processed
        low_res_output = self.process(img, filter_parameters)

        if high_res is not None:
            if self.no_high_res():
                high_res_output = high_res
            else:
                self.high_res_mask = self.get_mask(high_res, mask_parameters)
                high_res_output = lerp(high_res,
                                       self.process(high_res, filter_parameters),
                                       self.high_res_mask)
        else:
            high_res_output = None

        return low_res_output, filter_parameters

class UsmFilter(Filters):
    def __init__(self, net, config):
        super().__init__(net, config)
        self.cfg = config
        self.short_name = 'UF'
        self.begin_filter_parameter = self.cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param):
        def make_gaussian_2d_kernel(sigma, dtype=tf.float32):
            radius = 12
            x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
            k = tf.exp(-0.5 * tf.square(x / sigma))
            k = k / tf.reduce_sum(k)
            return tf.expand_dims(k, 1) * k

        kernel_i = make_gaussian_2d_kernel(5)
        print('kernel_i.shape', kernel_i.shape)
        kernel_i = tf.tile(kernel_i[:, :, tf.newaxis, tf.newaxis], [1, 1, 1, 1])

        pad_w = (25 - 1) // 2
        padded = tf.pad(img, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')
        outputs = []
        for channel_idx in range(3):
            data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
            data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], 'VALID')
            outputs.append(data_c)

        output = torch.concat(outputs, dim=3)
        img_out = (img - output) * param[:, None, None, :] + img

        return img_out


