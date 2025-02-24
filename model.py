import time
from tqdm import tqdm
import shutil
import os
import numpy as np
import cv2
import random
import tensorflow.contrib.layers as ly
from util_filters import *
from config_lowlight import cfg, args


# utils
def image_preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=0.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def write_mes(msg, log_name=None, show=True, mode='a'):
    get_end = lambda line: '' if line.endswith('\n') else '\n'
    if show:
        if isinstance(msg, str):
            print(msg, end=get_end(msg))
        elif isinstance(msg, (list, tuple)):
            for line in msg:
                print(line, end=get_end(line))  # might be a different thing
        else:
            print(msg)

    if log_name is not None:
        with open(log_name, mode) as f:
            f.writelines(msg)

# common
def extract_parameters_2(net, cfg, trainable):
    output_dim = cfg.num_filter_parameters
    # net = net - 0.5
    min_feature_map_size = 4
    print('extract_parameters_2 CNN:')
    channels = 16
    print('    ', str(net.get_shape()))
    net = convolutional(net, filters_shape=(3, 3, 3, channels), trainable=trainable, name='ex_conv0',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, channels, 2 * channels), trainable=trainable, name='ex_conv1',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable, name='ex_conv2',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable, name='ex_conv3',
                        downsample=True, activate=True, bn=False)
    net = convolutional(net, filters_shape=(3, 3, 2 * channels, 2 * channels), trainable=trainable, name='ex_conv4',
                        downsample=True, activate=True, bn=False)
    net = tf.reshape(net, [-1, 2048])
    features = ly.fully_connected(
        net,
        64,
        scope='fc1',
        activation_fn=lrelu,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    filter_features = ly.fully_connected(
        features,
        output_dim,
        scope='fc2',
        activation_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer())
    return filter_features


def convolutional(input_data, filters_shape, trainable, name, downsample=False, activate=True, bn=True):
    with tf.compat.v1.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"

        weight = tf.compat.v1.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                           shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.01))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)

        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.compat.v1.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                             dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)

        if activate == True: conv = tf.nn.leaky_relu(conv, alpha=0.1)

    return conv


def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]

    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.compat.v1.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))

    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())

    return output


def residual_block(input_data, input_channel, filter_num1, filter_num2, trainable, name):
    short_cut = input_data

    with tf.variable_scope(name):
        input_data = convolutional(input_data, filters_shape=(1, 1, input_channel, filter_num1),
                                   trainable=trainable, name='conv1')
        input_data = convolutional(input_data, filters_shape=(3, 3, filter_num1, filter_num2),
                                   trainable=trainable, name='conv2')

        residual_output = input_data + short_cut

    return residual_output


# backbone
def darknet53(input_data, trainable):
    with tf.variable_scope('darknet'):

        input_data = convolutional(input_data, filters_shape=(3, 3, 3, 32), trainable=trainable, name='conv0')
        input_data = convolutional(input_data, filters_shape=(3, 3, 32, 64),
                                   trainable=trainable, name='conv1', downsample=True)

        for i in range(1):
            input_data = residual_block(input_data, 64, 32, 64, trainable=trainable, name='residual%d' % (i + 0))

        input_data = convolutional(input_data, filters_shape=(3, 3, 64, 128),
                                   trainable=trainable, name='conv4', downsample=True)

        for i in range(2):
            input_data = residual_block(input_data, 128, 64, 128, trainable=trainable, name='residual%d' % (i + 1))

        input_data = convolutional(input_data, filters_shape=(3, 3, 128, 256),
                                   trainable=trainable, name='conv9', downsample=True)

        for i in range(8):
            input_data = residual_block(input_data, 256, 128, 256, trainable=trainable, name='residual%d' % (i + 3))

        route_1 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 256, 512),
                                   trainable=trainable, name='conv26', downsample=True)

        for i in range(8):
            input_data = residual_block(input_data, 512, 256, 512, trainable=trainable, name='residual%d' % (i + 11))

        route_2 = input_data
        input_data = convolutional(input_data, filters_shape=(3, 3, 512, 1024),
                                   trainable=trainable, name='conv43', downsample=True)

        for i in range(4):
            input_data = residual_block(input_data, 1024, 512, 1024, trainable=trainable, name='residual%d' % (i + 19))

        return route_1, route_2, input_data


# YOLOv3
class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable, input_data_clean):

        self.trainable = trainable
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        self.num_class = len(self.classes)
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.anchors = get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD
        self.isp_flag = cfg.YOLO.ISP_FLAG

        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.recovery_loss = self.__build_nework(input_data,
                                                                                                        self.isp_flag,
                                                                                                        input_data_clean)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data, isp_flag, input_data_clean):

        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []

        if isp_flag:
            with tf.variable_scope('extract_parameters_2'):
                input_data = tf.image.resize(input_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                filter_features = extract_parameters_2(input_data, cfg, self.trainable)

            # filter_features = tf.random_normal([1, 10], 0.5, 0.1)

            filters = cfg.filters
            #不是串联
            filters = [x(input_data, cfg) for x in filters]
            filter_parameters = []

            for j, filter in enumerate(filters):
                with tf.variable_scope('filter_%d' % j):
                    print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
                          filter.get_short_name())
                    print('      filter_features:', filter_features.shape)

                    # 这里每个滤波器是离散的，都是对input_process进行处理
                    filtered_image_batch, filter_parameter = filter.apply(
                        filtered_image_batch, filter_features)
                    filter_parameters.append(filter_parameter)
                    filter_imgs_series.append(filtered_image_batch)

                    print('      output:', filtered_image_batch.shape)
            self.filter_params = filter_parameters
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series

        recovery_loss = tf.reduce_sum(tf.pow(filtered_image_batch - input_data_clean, 2.0))  # /(2.0 * batch_size)

        # 网络部分
        input_data = filtered_image_batch
        route_1, route_2, input_data = darknet53(input_data, self.trainable)

        input_data = convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv52')
        input_data = convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv53')
        input_data = convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv54')
        input_data = convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv55')
        input_data = convolutional(input_data, (1, 1, 1024, 512), self.trainable, 'conv56')

        conv_lobj_branch = convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (self.num_class + 5)),
                                   trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv57')
        input_data = upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = convolutional(input_data, (3, 3, 256, 512), self.trainable, name='conv_mobj_branch')
        conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                   trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                   trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox, recovery_loss

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]

        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)
        conf_focal = self.focal(respond_bbox, pred_conf)
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[2], stride=self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        with tf.name_scope('recovery_loss'):
            recovery_loss = self.recovery_loss

        return giou_loss, conf_loss, prob_loss, recovery_loss


# dataset_lowlight
class Dataset(object):
    """implement Dataset here"""

    def __init__(self, dataset_type):
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        print('###################the total image:', len(annotations))
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        # 这里是真正循环应用的代码
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                        bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                # 取数据是的返回值
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                    batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))
        # print('*****************read image***************************')
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size],
                                         np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs


# trian_lowlight
if args.use_gpu == 0:
    gpu_id = '-1'
else:
    gpu_id = args.gpu_id
    gpu_list = list()
    gpu_ids = gpu_id.split(',')
    for i in range(len(gpu_ids)):
        gpu_list.append('/gpu:%d' % int(i))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))

set_ckpt_dir = args.ckpt_dir
args.ckpt_dir = os.path.join(exp_folder, set_ckpt_dir)
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

config_log = os.path.join(exp_folder, 'config.txt')
arg_dict = args.__dict__
msg = ['{}: {}\n'.format(k, v) for k, v in arg_dict.items()]
write_mes(msg, config_log, mode='w')


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT

        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"

        self.trainset = Dataset('train')
        self.testset = Dataset('test')

        self.steps_per_period = len(self.trainset)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('define_input'):
            self.input_data = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')
            self.label_sbbox = tf.compat.v1.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.compat.v1.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.compat.v1.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.compat.v1.placeholder(dtype=tf.float32, name='lbboxes')
            self.input_data_clean = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_data')

            self.trainable = tf.compat.v1.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            # 初始化模型
            self.model = YOLOV3(self.input_data, self.trainable, self.input_data_clean)
            t_variables = tf.trainable_variables()
            print("t_variables", t_variables)
            # self.net_var = [v for v in t_variables if not 'extract_parameters' in v.name]
            self.net_var = tf.compat.v1.global_variables()
            # 计算损失
            self.giou_loss, self.conf_loss, self.prob_loss, self.recovery_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            # self.loss only includes the detection loss.
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.compat.v1.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                               var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.compat.v1.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                                var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        # 定义一个无操作
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.compat.v1.train.Saver(self.net_var)
            self.saver = tf.compat.v1.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope('summary'):
            tf.compat.v1.summary.scalar("learn_rate", self.learn_rate)
            tf.compat.v1.summary.scalar("giou_loss", self.giou_loss)
            tf.compat.v1.summary.scalar("conf_loss", self.conf_loss)
            tf.compat.v1.summary.scalar("prob_loss", self.prob_loss)
            tf.compat.v1.summary.scalar("recovery_loss", self.recovery_loss)
            tf.compat.v1.summary.scalar("total_loss", self.loss)

            # logdir = "./data/log/"
            logdir = os.path.join(exp_folder, 'log')

            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.compat.v1.summary.merge_all()
            self.summary_writer = tf.compat.v1.summary.FileWriter(logdir, graph=self.sess.graph)

    def train(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                if args.lowlight_FLAG:
                    # lowlight_param = 1
                    # 2/3的概率添加处理
                    # if random.randint(0, 2) > 0:
                    # 添加低光照的参数
                    # lowlight_param = random.uniform(1.5, 5)
                    lowlight_param = random.uniform(5, 10)
                    _, summary, train_step_loss, train_step_loss_recovery, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.recovery_loss, self.global_step], feed_dict={
                            # 背低光处理的图像
                            self.input_data: np.power(train_data[0], lowlight_param),
                            # train_data[0]*np.exp(lowlight_param*np.log(2)),

                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            # 清晰的图像
                            self.input_data_clean: train_data[0],
                            self.trainable: True,
                        })
                else:
                    _, summary, train_step_loss, global_step_val = self.sess.run(
                        [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                            self.input_data: train_data[0],
                            self.label_sbbox: train_data[1],
                            self.label_mbbox: train_data[2],
                            self.label_lbbox: train_data[3],
                            self.true_sbboxes: train_data[4],
                            self.true_mbboxes: train_data[5],
                            self.true_lbboxes: train_data[6],
                            # 清晰的图像
                            self.input_data_clean: train_data[0],
                            self.trainable: True,
                        })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)

                pbar.set_description("train loss: %.2f" % (train_step_loss))

            if args.lowlight_FLAG:
                for test_data in self.testset:
                    # lowlight_param = random.uniform(-2, 0)
                    lowlight_param = 1
                    if random.randint(0, 2) > 0:
                        lowlight_param = random.uniform(5, 10)
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: np.power(test_data[0], lowlight_param),
                        # test_data[0]*np.exp(lowlight_param*np.log(2)),
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[0],
                        self.trainable: False,
                    })

                    test_epoch_loss.append(test_step_loss)
            else:
                for test_data in self.testset:
                    test_step_loss = self.sess.run(self.loss, feed_dict={
                        self.input_data: test_data[0],
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.input_data_clean: test_data[0],
                        self.trainable: False,
                    })

                    test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = args.ckpt_dir + "/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    YoloTrain().train()
