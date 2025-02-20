import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import argparse
import numpy as np
import cv2
from modelscope.models.cv.video_multi_object_tracking.utils.utils import cfg_opt


parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_num', dest='exp_num', type=str,
                    default='58', help='current experiment number')
parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int,
                    default=0, help='# of epochs')
parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int,
                    default=70, help='# of epochs')
parser.add_argument('--use_gpu', dest='use_gpu', type=int,
                    default=0, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir',
                    default='checkpoint', help='models are saved here')
parser.add_argument('--exp_dir', dest='exp_dir',
                    default='./experiments_lowlight', help='models are saved here')
parser.add_argument('--gpu_id', dest='gpu_id', type=str,
                    default='5', help='if use gpu, use gpu device id')
parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool,
                    default=True, help='whether use isp')
parser.add_argument('--lowlight_FLAG', dest='lowlight_FLAG', type=bool, default=True,
                    help='whether use Hybrid data training')
parser.add_argument('--train_path', dest='train_path', nargs='*',
                    default=r'D:\project\Image-Adaptive-YOLO-main\data\dataset_dark\tielu_train.txt',
                    help='folder of the training data')
parser.add_argument('--test_path', dest='test_path', nargs='*',
                    default=r'D:\project\Image-Adaptive-YOLO-main\data\dataset_dark\tielu_test.txt',
                    help='folder of the training data')
parser.add_argument('--class_name', dest='class_name', nargs='*',
                    default=r'D:\project\Image-Adaptive-YOLO-main\data\classes\tielu_dark.names',
                    help='folder of the training data')
parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', nargs='*',
                    default='./experiments_lowlight/exp_58/detection_vocnorm_test/', help='folder of the training data')
parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', nargs='*',
                    default='./experiments_lowlight/exp_58/checkpoint/yolov3_test_loss=25.2496.ckpt-15',
                    help='folder of the training data')
parser.add_argument('--pre_train', dest='pre_train',
                    default='NULL', help='the path of pretrained models if is not null. not used for now')



# 优化字典方法
class Dict(dict):
    """
      Example:
      m = Dict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Dict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Dict, self).__delitem__(key)
        del self.__dict__[key]


# 激活函数
def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * torch.abs(x)


# 转化亮度图片
def rgb2lum(image):
    lum = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
    # 添加一个通道维度，使输出形状为 (batch_size, 1, height, width)
    return lum.unsqueeze(1)


# 新激活函数tanh01
def tanh01(x):
    return torch.tanh(x) * 0.5 + 0.5


# 生成一个新的激活函数
def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):

        def activation(x):
            if initial is not None:
                bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else:
                bias = 0
            return tanh01(x + bias) * (right - left) + left

        return activation

    return get_activation(l, r, initial)


# 插值函数
def lerp(a, b, l):
    return (1 - l) * a + l * b


# 定义超类
class Filter:
    def __init__(self, cfg):
        self.cfg = cfg

        self.short_name = None
        self.num_filter_parameters = None
        self.filter_parameters = None

    def get_short_name(self):
        assert self.short_name is not None, 'short_name should not be None'
        return self.short_name

    def get_num_filter_parameters(self):
        assert self.num_filter_parameters is not None, 'filter_parameters should not be None'
        return self.num_filter_parameters

    def get_begin_filter_parameters(self):
        return self.begin_filter_parameter

    def extract_parameters(self, feature):
        return (feature[:, self.get_begin_filter_parameters():(self.get_begin_filter_parameters() + self.get_num_filter_parameters())],
                feature[:, self.get_begin_filter_parameters():(self.get_begin_filter_parameters() + self.get_num_filter_parameters())])

    # 强制子类的完成, 需要子类实现
    def filter_param_regressor(self, feature):
        assert False

    def process(self, img, param):
        assert False

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
            mask_parameters = torch.zeros(
                (1, self.get_num_mask_parameters()), dtype=torch.float32, device=img.device)

        debug_info = {}

        if self.debug_info_batched():
            debug_info['filter_parameters'] = filter_parameters
        else:
            debug_info['filter_parameters'] = filter_parameters[0]

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

    def use_masking(self):
        return self.cfg.masking

    def get_num_mask_parameters(self):
        return 6

    def get_mask(self, img, mask_parameters):
        if not self.use_masking():
            print('* Masking Disabled')
            return torch.ones((1, 1, 1, 1), dtype=torch.float32, device=img.device)
        else:
            print('* Masking Enabled')

        filter_input_range = 5
        assert mask_parameters.shape[1] == self.get_num_mask_parameters()

        # Convert mask parameters with tanh range
        mask_parameters = tanh_range(
            l=-filter_input_range, r=filter_input_range, initial=0)(mask_parameters)

        size = img.shape[2:]  # (N, C, H, W)
        shorter_edge = min(size[0], size[1])

        # Create grid using PyTorch
        i, j = torch.meshgrid(
            torch.arange(size[0], dtype=torch.float32, device=img.device),
            torch.arange(size[1], dtype=torch.float32, device=img.device),
            indexing='ij'
        )

        grid_i = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
        grid_j = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
        grid = torch.stack([grid_i, grid_j], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # Expand dimensions for broadcasting
        mask_params = mask_parameters.view(-1, 1, 1, self.get_num_mask_parameters())

        # Calculate input features
        lum = rgb2lum(img).permute(0, 2, 3, 1) - 0.5  # (N, H, W, C)
        inp = (
                grid[..., 0:1] * mask_params[..., 0:1] +
                grid[..., 1:2] * mask_params[..., 1:2] +
                mask_params[..., 2:3] * lum +
                mask_params[..., 3:4] * 2
        )

        # Apply sharpness
        sharpness = self.cfg.maximum_sharpness * mask_params[..., 4:5] / float(filter_input_range)
        inp *= sharpness

        # Sigmoid activation
        mask = torch.sigmoid(inp)

        # Apply strength
        strength = (mask_params[..., 5:6] / float(filter_input_range) * 0.5 + 0.5) * (
                    1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
        mask = mask * strength

        # Adjust dimensions for PyTorch (N, C, H, W)
        mask = mask.permute(0, 3, 1, 2)

        print('mask shape:', mask.shape)

        return mask

    def visualize_mask(self, debug_info, res):
        return cv2.resize(
            debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
            dsize=res,
            interpolation=cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(
            canvas,
            text, (30, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 0),
            thickness=5)
        return canvas


class UsmFilter(Filter):  # Usm_param is in [Defog_range]

    def __init__(self, net, cfg):
        Filter.__init__(net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param):
        def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=dtype, device=img.device)
            k = torch.exp(-0.5 * torch.square(x / sigma))
            k = k / k.sum()
            return torch.outer(k, k)  # 生成2D高斯核

        # 创建卷积核 (H, W)
        kernel = make_gaussian_2d_kernel(5)

        # 调整卷积核维度 (out_channels, in_channels, H, W)
        kernel = kernel.view(1, 1, *kernel.shape).repeat(1, 1, 1, 1)

        # 输入维度处理 (N, C, H, W)
        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')

        outputs = []
        # 拆分通道处理
        for channel in torch.unbind(padded, dim=1):
            # 添加通道维度 (N, 1, H, W)
            channel = channel.unsqueeze(1)

            # 执行卷积
            conv_output = F.conv2d(
                channel,
                kernel,
                stride=1,
                padding=0,
                groups=1  # 每个通道单独处理
            )
            outputs.append(conv_output)

        # 合并通道 (N, C, H, W)
        output = torch.cat(outputs, dim=1)

        # 调整param维度 (N, 1, 1, 1) -> (N, 1, H, W)
        param = param.view(-1, 1, 1, 1).expand_as(img)

        # 应用USM公式
        img_out = (img - output) * param + img

        return img_out

class ImprovedWhiteBalanceFilter:
    def __init__(self, net, cfg):
        self.net = net
        self.cfg = cfg
        self.short_name = 'W'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([0, 1, 1], dtype=torch.float32).reshape(1, 3)
        # mask = torch.tensor([1, 0, 1], dtype=torch.float32).reshape(1, 3)

        print(mask.shape)
        assert mask.shape == (1, 3)
        features = features * mask
        color_scaling = torch.exp(torch.tanh(features) * log_wb_range)
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        luminance = 1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2]
        color_scaling = color_scaling / luminance[:, None]
        return color_scaling

    def process(self, img, param):
        return img * param[:, None, None, :]
