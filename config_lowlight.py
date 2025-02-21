#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict
import argparse
import tensorflow as tf
import numpy as np
from util_filters import rgb2lum, tanh_range, lerp
import cv2
import math


parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_num', dest='exp_num', type=str, default='58', help='current experiment number')
parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int, default=0, help='# of epochs')
parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int, default=70, help='# of epochs')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=0, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint', help='models are saved here')
parser.add_argument('--exp_dir', dest='exp_dir', default='./experiments_lowlight', help='models are saved here')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='5', help='if use gpu, use gpu device id')
parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use isp')
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



# filter_lowlight
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

        return (features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())],
                features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())])

    # Should be implemented in child classes
    def filter_param_regressor(self, features):
        assert False

    # Process the whole image, without masking
    # Should be implemented in child classes
    def process(self, img, param):
        assert False

    def debug_info_batched(self):
        return False

    def no_high_res(self):
        return False

    # Apply the whole filter with masking
    def apply(self,img,img_features=None,specified_parameter=None,high_res=None):
        assert (img_features is None) ^ (specified_parameter is None)
        if img_features is not None:
            filter_features, mask_parameters = self.extract_parameters(img_features)
            filter_parameters = self.filter_param_regressor(filter_features)
        else:
            assert not self.use_masking()
            filter_parameters = specified_parameter
            mask_parameters = tf.zeros(
                shape=(1, self.get_num_mask_parameters()), dtype=np.float32)
        if high_res is not None:
            # working on high res...
            pass
        debug_info = {}
        # We only debug the first image of this batch
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
            return tf.ones(shape=(1, 1, 1, 1), dtype=tf.float32)
        else:
            print('* Masking Enabled')
        with tf.name_scope(name='mask'):
            # Six parameters for one filter
            filter_input_range = 5
            assert mask_parameters.shape[1] == self.get_num_mask_parameters()
            mask_parameters = tanh_range(
                l=-filter_input_range, r=filter_input_range,
                initial=0)(mask_parameters)
            size = list(map(int, img.shape[1:3]))
            grid = np.zeros(shape=[1] + size + [2], dtype=np.float32)

            shorter_edge = min(size[0], size[1])
            for i in range(size[0]):
                for j in range(size[1]):
                    grid[0, i, j,
                    0] = (i + (shorter_edge - size[0]) / 2.0) / shorter_edge - 0.5
                    grid[0, i, j,
                    1] = (j + (shorter_edge - size[1]) / 2.0) / shorter_edge - 0.5
            grid = tf.constant(grid)
            # Ax + By + C * L + D
            inp = grid[:, :, :, 0, None] * mask_parameters[:, None, None, 0, None] + \
                  grid[:, :, :, 1, None] * mask_parameters[:, None, None, 1, None] + \
                  mask_parameters[:, None, None, 2, None] * (rgb2lum(img) - 0.5) + \
                  mask_parameters[:, None, None, 3, None] * 2
            # Sharpness and inversion
            inp *= self.cfg.maximum_sharpness * mask_parameters[:, None, None, 4,
                                                None] / filter_input_range
            mask = tf.sigmoid(inp)
            # Strength
            mask = mask * (
                    mask_parameters[:, None, None, 5, None] / filter_input_range * 0.5 +
                    0.5) * (1 - self.cfg.minimum_strength) + self.cfg.minimum_strength
            print('mask', mask.shape)
        return mask

    # def visualize_filter(self, debug_info, canvas):
    #   # Visualize only the filter information
    #   assert False

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
        Filter.__init__(self, net, cfg)
        self.short_name = 'UF'
        self.begin_filter_parameter = cfg.usm_begin_param
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

        output = tf.concat(outputs, axis=3)
        img_out = (img - output) * param[:, None, None, :] + img

        return img_out


class ImprovedWhiteBalanceFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
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
        color_scaling = tf.exp(tanh_range(-log_wb_range, log_wb_range)(features))
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling *= 1.0 / (
                                       1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
                                       0.06 * color_scaling[:, 2])[:, None]
        return color_scaling

    def process(self, img, param):
        return img * param[:, None, None, :]


class GammaFilter(Filter):  # gamma_param is in [1/gamma_range, gamma_range]

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = np.log(self.cfg.gamma_range)
        return tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

    def process(self, img, param):
        param_1 = tf.tile(param, [1, 3])
        return tf.pow(tf.maximum(img, 0.001), param_1[:, None, None, :])

class ToneFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.curve_steps = cfg.curve_steps
        self.short_name = 'T'
        self.begin_filter_parameter = cfg.tone_begin_param

        self.num_filter_parameters = cfg.curve_steps

    def filter_param_regressor(self, features):
        tone_curve = tf.reshape(
            features, shape=(-1, 1, self.cfg.curve_steps))[:, None, None, :]
        tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
        return tone_curve

    def process(self, img, param):
        # img = tf.minimum(img, 1.0)
        tone_curve = param
        tone_curve_sum = tf.reduce_sum(tone_curve, axis=4) + 1e-30
        total_image = img * 0
        for i in range(self.cfg.curve_steps):
            total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                           * param[:, :, :, :, i]
        total_image *= self.cfg.curve_steps / tone_curve_sum
        img = total_image
        return img


class ContrastFilter(Filter):

    def __init__(self, net, cfg):
        Filter.__init__(self, net, cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param

        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        # return tf.sigmoid(features)
        # return tanh_range(*self.cfg.contrast_range)(features)

        return tf.tanh(features)

    def process(self, img, param):
        luminance = tf.minimum(tf.maximum(rgb2lum(img), 0.0), 1.0)
        contrast_lum = -tf.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, param[:, :, None, None])



args = parser.parse_args()

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

###########################################################################
# Filter Parameters
###########################################################################


cfg.filters = [ImprovedWhiteBalanceFilter, GammaFilter,
               ToneFilter, ContrastFilter, UsmFilter
               ]
cfg.num_filter_parameters = 14

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

# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

###########################################################################
# CNN Parameters
###########################################################################
cfg.source_img_size = 64
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096

###########################################################################

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES = args.class_name
__C.YOLO.ANCHORS = r"D:\project\Image-Adaptive-YOLO-main\data\anchors\baseline_anchors.txt"
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
__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.DATA_AUG = True
# __C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_INIT = 1e-2
__C.TRAIN.LEARN_RATE_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FISRT_STAGE_EPOCHS = args.epoch_first_stage
__C.TRAIN.SECOND_STAGE_EPOCHS = args.epoch_second_stage
__C.TRAIN.INITIAL_WEIGHT = args.pre_train

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = args.test_path
__C.TEST.BATCH_SIZE = 6
__C.TEST.INPUT_SIZE = 544
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = args.WRITE_IMAGE_PATH
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = args.WEIGHT_FILE
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
