import argparse
import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
import cv2
import math

# Command line arguments
parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_num', dest='exp_num', type=str, default='58', help='current experiment number')
parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int, default=0, help='not set zero if use pre_train')
parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int, default=70, help='all of the epochs')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--exp_dir', dest='exp_dir', default='./experiments_lowlight', help='models are saved here')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='0', help='if use gpu, use gpu device id')
parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use isp')
parser.add_argument('--lowlight_FLAG', dest='lowlight_FLAG', type=bool, default=True,
                    help='whether use Hybrid data training')
parser.add_argument('--train_path', dest='train_path', nargs='*',
                    default='/home/youtian/Documents/pro/pyCode/Dedark-YOLO/pytorch_file/data/tielu_train.txt',
                    help='folder of the training data')
parser.add_argument('--test_path', dest='test_path', nargs='*',
                    default='/home/youtian/Documents/pro/pyCode/Dedark-YOLO/pytorch_file/data/tielu_test.txt',
                    help='folder of the training data')
parser.add_argument('--class_name', dest='class_name', nargs='*',
                    default='/home/youtian/Documents/pro/pyCode/Dedark-YOLO/pytorch_file/data/tielu_dark.names',
                    help='folder of the training data')
parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', nargs='*',
                    default='./experiments_lowlight/exp_58/detection_vocnorm_test/', help='folder of the training data')
parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', nargs='*',
                    default='./experiments_lowlight/exp_58/checkpoint/yolov3_test_loss=25.2496.ckpt-15',
                    help='folder of the training data')
parser.add_argument('--pre_train', dest='pre_train',
                    default='NULL', help='the path of pretrained models if is not null. not used for now')
args = parser.parse_args()

# Configuration
__C = edict()
cfg = __C

# Filter Parameters
cfg.filters = ['ImprovedWhiteBalanceFilter', 'GammaFilter', 'ToneFilter', 'ContrastFilter', 'UsmFilter']
cfg.num_filter_parameters = 14
cfg.device = "cuda" if args.use_gpu == 1 else "cpu"
cfg.wb_begin_param = 0
cfg.gamma_begin_param = 3
cfg.tone_begin_param = 4
cfg.contrast_begin_param = 12
cfg.usm_begin_param = 13
cfg.curve_steps = 4
cfg.gamma_range = 2.5
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.5, 1.0)
cfg.usm_range = (0.0, 2.5)
cfg.contrast_range = (0.0, 1.0)
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

# CNN Parameters
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
cfg.feature_extractor_dims = 4096

# YOLO options
__C.YOLO = edict()
__C.YOLO.CLASSES = args.class_name
__C.YOLO.ANCHORS = "/home/youtian/Documents/pro/pyCode/Dedark-YOLO/pytorch_file/data/baseline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.ISP_FLAG = args.ISP_FLAG

# Train options
__C.TRAIN = edict()
__C.TRAIN.ANNOT_PATH = args.train_path
__C.TRAIN.BATCH_SIZE = 6
# __C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = 416
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FISRT_STAGE_EPOCHS = args.epoch_first_stage
__C.TRAIN.SECOND_STAGE_EPOCHS = args.epoch_second_stage
__C.TRAIN.INITIAL_WEIGHT = args.pre_train

# TEST options
__C.TEST = edict()
__C.TEST.ANNOT_PATH = args.test_path
__C.TEST.BATCH_SIZE = 6
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = args.WRITE_IMAGE_PATH
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = args.WEIGHT_FILE
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45


# Helper functions
def rgb2lum(img):
    return 0.27 * img[:, 0] + 0.67 * img[:, 1] + 0.06 * img[:, 2]


def tanh_range(l, r, initial=0):
    def func(features):
        return torch.tanh(features) * (r - l) / 2 + (r + l) / 2

    return func


def lerp(a, b, t):
    return a + (b - a) * t


# Filter base class
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
            mask_parameters = torch.zeros(1, self.get_num_mask_parameters(), dtype=torch.float32)
        if high_res is not None:
            pass
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
            return torch.ones(1, 1, 1, 1, dtype=torch.float32)
        else:
            print('* Masking Enabled')
        filter_input_range = 5
        assert mask_parameters.shape[1] == self.get_num_mask_parameters()
        mask_parameters = tanh_range(-filter_input_range, filter_input_range, initial=0)(mask_parameters)
        size = list(map(int, img.shape[1:3]))
        grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

        shorter_edge = min(size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                grid[0, i, j, 0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                grid[0, i, j, 1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
        grid = torch.tensor(grid, dtype=torch.float32)
        inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
              grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
              mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
              mask_parameters[:, None, None, 3, None] * 2
        inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4, None] / filter_input_range
        mask = torch.sigmoid(inp)
        mask = mask * (mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 + 0.5) * (
                    1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
        return mask

    def visualize_mask(self, debug_info, res):
        return cv2.resize(debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32), dsize=res,
                          interpolation=cv2.INTER_NEAREST)

    def draw_high_res_text(self, text, canvas):
        cv2.putText(canvas, text, (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=5)
        return canvas


# Improved White Balance Filter
class ImprovedWhiteBalanceFilter(Filter):
    def __init__(self, net, cfg):
        super(ImprovedWhiteBalanceFilter, self).__init__(net, cfg)
        self.short_name = 'W'
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = 3

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        mask = torch.tensor([0, 1, 1], dtype=torch.float32).reshape(1, 3)
        features = features * mask
        color_scaling = torch.exp(torch.tanh(features) * log_wb_range)
        luminance = 1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] + 0.06 * color_scaling[:, 2]
        color_scaling = color_scaling / luminance[:, None]
        return color_scaling

    def process(self, img, param):
        return img * param[:, None, None, :]


# Gamma Filter
class GammaFilter(Filter):
    def __init__(self, net, cfg):
        super(GammaFilter, self).__init__(net, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.cfg.gamma_range)
        return torch.exp(torch.tanh(features) * log_gamma_range)

    def process(self, img, param):
        param_1 = param.repeat(1, 3)
        return torch.pow(torch.clamp(img, min=0.001), param_1[:, None, None, :])


# Tone Filter
class ToneFilter(Filter):
    def __init__(self, net, cfg):
        super(ToneFilter, self).__init__(net, cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param
        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = features.view(-1, 1, self.cfg.curve_steps)
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param):
        tone_curve = param
        tone_curve_sum = torch.sum(tone_curve, dim=4) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) * param[:,
                                                                                                              :, :, :,
                                                                                                              i]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        return total_image


# Contrast Filter
class ContrastFilter(Filter):
    def __init__(self, net, cfg):
        super(ContrastFilter, self).__init__(net, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return torch.tanh(features)

    def process(self, img, param):
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])


# Usm Filter
class UsmFilter(Filter):
    def __init__(self, net, cfg):
        super(UsmFilter, self).__init__(net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        return tanh_range(*self.cfg.usm_range)(features)

    def process(self, img, param):
        def make_gaussian_2d_kernel(sigma):
            radius = 12
            x = torch.arange(-radius, radius + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * torch.square(x / sigma))
            k = k / torch.sum(k)
            return k[:, None] * k[None, :]

        kernel_i = make_gaussian_2d_kernel(5)
        kernel_i = kernel_i[None, None, :, :].repeat(1, 3, 1, 1)
        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        output = F.conv2d(padded, kernel_i, groups=3)
        img_out = (img - output) * param[:, None, None, :] + img
        return img_out
