from pathlib import Path
import cv2
import os
import shutil
import numpy as np
import torch
import core.utils as utils
from core.config_lowlight import cfg
from core.yolov3_lowlight import YOLOV3
from core.config_lowlight import args
import time


exp_folder = os.path.join(args.exp_dir, 'exp_{}'.format(args.exp_num))
device = "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloTest(object):
    def __init__(self):
        self.input_size = cfg.TEST.INPUT_SIZE
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        # 测试文件的txt文件
        self.annotation_path = cfg.TEST.ANNOT_PATH
        # 加载的权重文件
        self.weight_file = cfg.TEST.WEIGHT_FILE
        # ture
        self.write_image = cfg.TEST.WRITE_IMAGE
        # 测试结果的文件路径
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH

        self.show_label = cfg.TEST.SHOW_LABEL
        self.isp_flag = cfg.YOLO.ISP_FLAG

        # 加载权重
        self.model = YOLOV3().to(device)
        self.model.load_state_dict(torch.load(self.weight_file))

    def predict(self, image):
        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape
        # 拿到归一化目标尺寸的图像
        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        # 添加一个bs纬度
        image_data = image_data[np.newaxis, ...]

        # 添加lowlight作为输入图片
        # lowlight_param = random.uniform(1.5, 5)

        # 不添加第光照参数
        lowlight_param = 1
        lowlight_data = np.power(image_data, lowlight_param)

        image_data = torch.from_numpy(image_data).to(device).to(torch.float32)
        lowlight_data = torch.from_numpy(lowlight_data).to(device).to(torch.float32)

        image_data = image_data.permute(0, 3, 1, 2)
        lowlight_data = lowlight_data.permute(0, 3, 1, 2)
        # 这里在使用的时候也需要使用清晰图像吗？
        pred_sbbox, pred_mbbox, pred_lbbox, image_isped, isp_param = self.model(lowlight_data, image_data)


        pred_bbox = np.concatenate([np.reshape(pred_sbbox.detach().cpu().numpy(), (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox.detach().cpu().numpy(), (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox.detach().cpu().numpy(), (-1, 5 + self.num_classes))], axis=0)

        # 使用后处理将特征恢复为原图的bbox
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)

        # nms
        bboxes = utils.nms(bboxes, self.iou_threshold)

        if self.isp_flag:
            # print('ISP params :  ', isp_param)
            image_isped = np.clip(image_isped[0].permute(1, 2, 0).detach().cpu().numpy() * 255, 0, 255)

            # save_dir = Path('runs/detect/exp/filtered')
            # save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
            # save_path = save_dir / f'filtered_img{1}.jpg'
            #
            # # 保存图像
            # cv2.imwrite(str(save_path), image_isped)
            # print(f"Saved filtered image to {save_path}")

            image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
        else:
            image_isped = np.clip(image, 0, 255)

            # 保留中间的图片
            # image_isped = utils.image_unpreporcess(image_isped, [org_h, org_w])
            # cv2.imwrite(self.write_image_path + 'low'+ image_name, image_isped)

        return bboxes, image_isped


    def evaluate_once(self,
                      img_path="/home/youtian/Documents/pro/pyCode/datasets/darkpic_test/192.168.39.20_20240727_060304_2246886_2.jpg",
                      save_dir="runs/detect/exp/filtered"):
        """
        评估单张图像
        :param img_path: 图像路径，如果为None则使用默认路径
        :param save_dir: 结果保存目录
        """
        # 确保保存目录存在
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 读取图像
        if not os.path.exists(img_path):
            print(f"错误：图像路径不存在 - {img_path}")
            return None

        img = cv2.imread(img_path)
        if img is None:
            print(f"错误：无法读取图像 - {img_path}")
            return None

        # 进行预测
        bboxes_pr, image_isped = self.predict(img)

        # 绘制边界框
        image = utils.draw_bbox(
            image_isped,
            bboxes_pr,
            self.classes,
            show_label=self.show_label
        )

        # 生成唯一文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = save_dir / f"filtered_img_{timestamp}.jpg"

        # 保存结果
        cv2.imwrite(str(save_path), image)
        print(f"结果已保存至: {save_path}")
        return save_path

    def evaluate(self):
        mAP_path = exp_folder + '/mAP'
        #
        if not os.path.exists(mAP_path):
            os.makedirs(mAP_path)

        predicted_dir_path = mAP_path + '/predicted'
        ground_truth_dir_path = mAP_path + '/ground-truth'

        if os.path.exists(predicted_dir_path):
            shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path):
            shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path):
            shutil.rmtree(self.write_image_path)

        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        time_total = 0
        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)

                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_path)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_path)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                # bboxes_pr, image_isped = self.predict(image, image_name)
                t1 = time.time()
                bboxes_pr, image_isped = self.predict(image)
                time_total += time.time() - t1

                if self.write_image:
                    if self.isp_flag:
                        image = utils.draw_bbox(image_isped, bboxes_pr, self.classes, show_label=self.show_label)
                    else:
                        image = utils.draw_bbox(image_isped, bboxes_pr, self.classes, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path + image_name, image)

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
                        print('\t' + str(bbox_mess).strip())





if __name__ == '__main__':
    # YoloTest().evaluate()
    YoloTest().evaluate_once()
