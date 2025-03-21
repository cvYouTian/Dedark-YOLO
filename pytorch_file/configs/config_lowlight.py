import argparse
import torch
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
import cv2
import math

parser = argparse.ArgumentParser(description='')
parser.add_argument('--exp_num', dest='exp_num', type=str, default='58', help='current experiment number')
parser.add_argument('--epoch_first_stage', dest='epoch_first_stage', type=int, default=20, help='# of epochs')
parser.add_argument('--epoch_second_stage', dest='epoch_second_stage', type=int, default=70, help='# of epochs')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='checkpoint', help='models are saved here')
# parser.add_argument('--exp_dir', dest='exp_dir', default='./experiments_lowlight', help='models are saved here')
parser.add_argument('--exp_dir', dest='exp_dir', default='./experiments_lowlight', help='models are saved here')
parser.add_argument('--gpu_id', dest='gpu_id', type=str, default='5', help='if use gpu, use gpu device id')
parser.add_argument('--ISP_FLAG', dest='ISP_FLAG', type=bool, default=True, help='whether use isp')
parser.add_argument('--lowlight_FLAG', dest='lowlight_FLAG', type=bool, default=True, help='whether use Hybrid data training')
# linux
parser.add_argument('--train_path', dest='train_path', nargs='*', default='/home/youtian/Documents/pro/pyCode/DarkYOLO/Image-Adaptive-YOLO-main/data/dataset_dark/tielu_train.txt', help='folder of the training data')
parser.add_argument('--test_path', dest='test_path', nargs='*', default='/home/youtian/Documents/pro/pyCode/DarkYOLO/Image-Adaptive-YOLO-main/data/dataset_dark/tielu_test.txt', help='folder of the training data')
parser.add_argument('--class_name', dest='class_name', nargs='*', default='/home/youtian/Documents/pro/pyCode/DarkYOLO/Image-Adaptive-YOLO-main/data/classes/tielu_dark.names', help='folder of the training data')
parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', nargs='*', default='./experiments_lowlight/exp_58/detection_test/', help='folder of the training data')
parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', nargs='*', default='./experiments_lowlight/exp_58/checkpoint/yolov3_test_loss=9.0280.ckpt-71', help='folder of the training data')
# win
# parser.add_argument('--train_path', dest='train_path', nargs='*', default=r'D:\project\Image-Adaptive-YOLO-main\data\dataset_dark\tielu_train.txt', help='folder of the training data')
# parser.add_argument('--test_path', dest='test_path', nargs='*', default=r'D:\project\Image-Adaptive-YOLO-main\data\dataset_dark\tielu_test.txt', help='folder of the training data')
# parser.add_argument('--class_name', dest='class_name', nargs='*', default=r'D:\project\Image-Adaptive-YOLO-main\data\classes\tielu_dark.names', help='folder of the training data')
# # 这里的路径后面要加入斜线
# parser.add_argument('--WRITE_IMAGE_PATH', dest='WRITE_IMAGE_PATH', nargs='*', default='D:\\project\\Image-Adaptive-YOLO-main\\experiments_lowlight\\exp_58\\detection_test\\', help='folder of the training data')
# parser.add_argument('--WEIGHT_FILE', dest='WEIGHT_FILE', nargs='*', default='D:\project\Image-Adaptive-YOLO-main\experiments_lowlight\exp_58\checkpoint\yolov3_test_loss=9.0280.ckpt-71', help='folder of the training data')


parser.add_argument('--pre_train', dest='pre_train', default='./experiments_lowlight/exp_58/checkpoint1/yolov3_test_loss=25.2496.ckpt-15', help='the path of pretrained models if is not null. not used for now')
# we trained our model from scratch.

args = parser.parse_args()


# Configuration
__C = edict()
cfg = __C

# Filter Parameters
cfg.filters = ['ImprovedWhiteBalanceFilter', 'GammaFilter', 'ContrastFilter', 'UsmFilter']
cfg.num_filter_parameters = 14
cfg.device = "cuda:0" if args.use_gpu == 1 else "cpu"
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
# win
# __C.YOLO.ANCHORS = "D:\\project\\Dedark-YOLO\\pytorch_file\\baseline_anchors.txt"
# linux
__C.YOLO.ANCHORS = "/home/youtian/Documents/pro/pyCode/DarkYOLO/Image-Adaptive-YOLO-main/data/anchors/baseline_anchors.txt"

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

