import cv2
import os
import shutil
import numpy as np
import torch
from tqdm import tqdm
import time
from core.yolov3_lowlight import YOLOV3
import core.utils as utils
from core.config_lowlight import cfg, args

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")


class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.annotation_path = cfg.TEST.ANNOT_PATH
        self.weight_file = cfg.TEST.WEIGHT_FILE
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL
        self.isp_flag = cfg.YOLO.ISP_FLAG

        # 初始化模型
        self.model = YOLOV3().to(device)

        # 加载权重
        if os.path.exists(self.weight_file):
            self.model.load_state_dict(torch.load(self.weight_file))
            print(f"Loaded weights from {self.weight_file}")
        else:
            raise ValueError(f"Weight file {self.weight_file} not found")

        # 设置为评估模式
        self.model.eval()

    def predict(self, image, image_name):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape

        # 图像预处理
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]  # 添加batch维度

        # 转换为torch张量
        image_tensor = torch.from_numpy(image_data).to(device).float()
        image_tensor = image_tensor.permute(0, 3, 1, 2)


        # 前向传播
        with torch.no_grad():
            pred_sbbox, pred_mbbox, pred_lbbox, image_isped, isp_param = self.model(image_tensor, image_tensor)

        # 处理预测结果
        pred_sbbox = pred_sbbox.detach().cpu().numpy()
        pred_mbbox = pred_mbbox.detach().cpu().numpy()
        pred_lbbox = pred_lbbox.detach().cpu().numpy()

        pred_bbox = np.concatenate([
            np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
            np.reshape(pred_lbbox, (-1, 5 + self.num_classes))
        ], axis=0)

        # 后处理
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        # 处理输出图像
        if self.isp_flag:
            # print('ISP params:', isp_param)
            image_isped = image_isped[0].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
            image_isped = np.clip(image_isped * 255, 0, 255).astype(np.uint8)
            image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
        else:
            image_isped = np.clip(org_image, 0, 255)

        return bboxes, image_isped

    def evaluate(self):
        exp_folder = os.path.join(args.exp_dir, f'exp_{args.exp_num}')
        mAP_path = os.path.join(exp_folder, 'mAP')

        # 创建目录
        os.makedirs(mAP_path, exist_ok=True)
        predicted_dir_path = os.path.join(mAP_path, 'predicted')
        ground_truth_dir_path = os.path.join(mAP_path, 'ground-truth')

        # 清理并创建目录
        shutil.rmtree(predicted_dir_path, ignore_errors=True)
        shutil.rmtree(ground_truth_dir_path, ignore_errors=True)
        shutil.rmtree(self.write_image_path, ignore_errors=True)

        os.makedirs(predicted_dir_path)
        os.makedirs(ground_truth_dir_path)
        os.makedirs(self.write_image_path)

        time_total = 0

        # 读取标注文件
        with open(self.annotation_path, 'r') as annotation_file:
            annotations = annotation_file.readlines()

        # 使用tqdm显示进度条
        for num, line in enumerate(tqdm(annotations, desc="Evaluating")):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)

            # 处理真实标注
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]

            # 保存真实标注
            ground_truth_path = os.path.join(ground_truth_dir_path, f'{num}.txt')
            with open(ground_truth_path, 'w') as f:
                for i in range(len(bboxes_gt)):
                    class_name = self.classes[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)

            # 预测
            t1 = time.time()
            bboxes_pr, image_isped = self.predict(image, image_name)
            time_total += time.time() - t1

            # 保存预测结果
            predict_result_path = os.path.join(predicted_dir_path, f'{num}.txt')
            with open(predict_result_path, 'w') as f:
                for bbox in bboxes_pr:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = self.classes[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)

            # 保存图像
            if self.write_image:
                image_draw = utils.draw_bbox(image_isped, bboxes_pr, self.classes, show_label=self.show_label)
                cv2.imwrite(os.path.join(self.write_image_path, image_name), image_draw)

        print(f'Total evaluation time: {time_total:.2f}s')


if __name__ == '__main__':
    YoloTest().evaluate()