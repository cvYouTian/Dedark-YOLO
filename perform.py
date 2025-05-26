import shutil
import sys
import os
import xml.etree.ElementTree as ET
import torch
from ultralytics.nn.modules import Detect, Segment, AsffDetect
from ultralytics.utils.downloads import download
from pathlib import Path
from typing import Union
import cv2
import netron
import time
from tqdm import tqdm
from ultralytics import YOLO, RTDETR


def train():
    # Load a model
    model = YOLO('yolov8l.yaml')
    # model = RTDETR('rtdetr-l.yaml')
    # print(model)
    # model = YOLO('yolov8m.pt')

    # 做预训练
    # model = YOLO('yolov8x.pt')
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # Train the model
    # model.train(data="HSTS6.yaml", epochs=150, imgsz=640)
    model.train(data="coco128.yaml", epochs=5, imgsz=640)


def onnx(path: Union[str, Path] = "/home/youtian/Documents/pro/pyCode/easy_YOLOv8/yolov8l.pt"):
    # you need numpy==1.24.3 ,otherwise it will report Error
    onnxpath = Path(path).with_suffix(".onnx")
    print(onnxpath)
    if not onnxpath.exists():
        model = YOLO(path)
        # Export the model
        model.export(format='onnx')
    try:
        netron.start(str(onnxpath))
    except Exception as e:
        print(e)


def test_img():
    # print(torch.load("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/RFB+ASFF/weights/best.pt"))
    model = YOLO("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/RFB+ASFF/weights/best.pt")
    img = cv2.imread("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/images/16182.jpg")
    res = model(img)
    ann = res[0].plot()
    while True:
        cv2.imshow("yolo", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cur_path = sys.path[0]
    print(cur_path, sys.path)

    cv2.imwrite(cur_path + os.sep + "out2.jpg", ann)


def test_video():
    model = YOLO("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/yolov8x.pt")
    path = Path("/home/youtian/Documents/pro/pyCode/datasets/shitang.mp4")
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 使用正确的文件名
    output_path = Path(f"{path.stem}_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 尝试使用XVID编码
    out = cv2.VideoWriter(str(output_path), fourcc, 40, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            res = model(frame)
            ann = res[0].plot(line_width=3)
            cv2.imshow("yolo", ann)
            out.write(ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()


def test_folders(model_path: str = "/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/YOLOv8l+CHST6+32/weights/best.pt",
                 srcpath: str = "/home/youtian/Documents/pro/pyCode/datasets/HSTS6/CHTS6/images/val",
                 mothed: str="YOLO") -> None:

    # 加载权重model
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    if mothed.lower() == "rtdetr":
        model = RTDETR(str(model_path))
    else:
        model = YOLO(model_path)

    src = Path(srcpath) if not isinstance(srcpath, Path) else srcpath
    dst_folder = Path(sys.path[0]) / Path(f"{model_path.parts[-3]}_val_test_pic")
    if dst_folder.exists():
        shutil.rmtree(dst_folder) if any(dst_folder.iterdir()) else dst_folder.rmdir()
    dst_folder.mkdir(exist_ok=True, parents=True)
    timer = 0
    for img_path in src.iterdir():
        start_timer = time.time()
        res = model(cv2.imread(str(img_path)))
        end_timer = time.time()
        timer += end_timer - start_timer

        img = res[0].plot()
        # 把测试的图片提前resize成相同的size
        ann = cv2.resize(img, (640, 640))
        cv2.imwrite(str(Path(dst_folder) / Path(img_path.name)), ann)

    # 计算每一张的推理时间
    print("test time : %f" % (timer / len(list(src.iterdir()))))


def tracker():
    pa = "/home/you/Downloads/l.mp4"
    cap = cv2.VideoCapture(pa)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),)
    # cap = cv2.VideoCapture("/home/you/Desktop/YOLOv8/easy_YOLOv8//sample.mp4")
    model = YOLO("/home/you/Desktop/YOLOv8/easy_YOLOv8/runs/detect/train3"
                 "/weights/best.pt")
    flag = 0
    out = cv2.VideoWriter('airplaneSort.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 40, size)
    while True:
        if flag < 1:
            flag += 1
            continue
        else:
            flag += 1
            ret, frame = cap.read()
            if not ret:
                break
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            # result.boxes.id.cpu().numpy().astype(int)
            try:
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                for box, id in zip(boxes, ids):
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Id {id}",
                        (box[0], box[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
            except Exception as e:
                print(e)
            cv2.imshow("frame", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


class VOCprocess:
    def __init__(self):
        # Download
        # dir = Path(yaml['path'])  # dataset root dir
        dir = Path("../datasets/VOC")  # dataset root dir
        url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'
        urls = [f'{url}VOCtrainval_06-Nov-2007.zip',  # 446MB, 5012 images
                f'{url}VOCtest_06-Nov-2007.zip',  # 438MB, 4953 images
                f'{url}VOCtrainval_11-May-2012.zip']  # 1.95GB, 17126 images
        download(urls, dir=dir / 'images', curl=True, threads=3)

        # Convert
        self.path = dir / 'images/VOCdevkit'

    def convert_label(self, path, lb_path, year, image_id):
        """
        convert VOCdataset into the dataset of the yolov8
        """
        def convert_box(size, box):
            dw, dh = 1. / size[0], 1. / size[1]
            x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
            return x * dw, y * dh, w * dw, h * dh

        in_file = open(path / f'VOC{year}/Annotations/{image_id}.xml')
        out_file = open(lb_path, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                 "dog", "horse", "motorbike", "person",
                 "pottedplant", "sheep", "sofa", "train", "tvmonitor"]  # names list

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls in names and int(obj.find('difficult').text) != 1:
                xmlbox = obj.find('bndbox')
                bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in ('xmin', 'xmax', 'ymin', 'ymax')])
                cls_id = names.index(cls)  # class id
                out_file.write(" ".join([str(a) for a in (cls_id, *bb)]) + '\n')

    def run(self):
        for year, image_set in ('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
            # for year, image_set in ('2007', 'train'), ('2007', 'val'), ('2007', 'test'):
            imgs_path = dir / 'images' / f'{image_set}{year}'
            lbs_path = dir / 'labels' / f'{image_set}{year}'
            imgs_path.mkdir(exist_ok=True, parents=True)
            lbs_path.mkdir(exist_ok=True, parents=True)

            with open(self.path / f'VOC{year}/ImageSets/Main/{image_set}.txt') as f:
                image_ids = f.read().strip().split()
            for id in tqdm(image_ids, desc=f'{image_set}{year}'):
                f = self.path / f'VOC{year}/JPEGImages/{id}.jpg'  # old img path
                lb_path = (lbs_path / f.name).with_suffix('.txt')  # new label path
                f.rename(imgs_path / f.name)  # move image
                self.convert_label(self.path, lb_path, year, id)  # convert labels to YOLO format


def videopin():
    from moviepy.editor import VideoFileClip, clips_array

    clip1 = VideoFileClip('our.mp4')  # 读入视频
    clip2 = VideoFileClip('our1.mp4')
    # final_clip = clips_array([[clip1], [clip2]])  # 上下拼接
    final_clip = clips_array([[clip1, clip2]])#左右拼接

    final_clip.write_videofile('c.mp4')  # 保存视频


def check_GPU():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def Para4pt(model):
    # 一定要找到权重中的model，才可以后续进行parameter的计算
    # YOLOv8
    # model = model["model"].model
    # YOLOv7
    model = model["model"].model

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    # print(f'{total_params / (1024 * 1024):.2f}M total parameters.')


def FLOPs_Para4pt():
    from thop import profile
    """
    根据pytorch的pt文件来计算模型的FLOPs和参数量，use thop
    Args:
        model:pytorch加载好的模型文件

    Returns:None
    """

    pa = Path("/home/youtian/Documents/pro/pyCode/ultralytics-YOLOv8/yolov8l.pt")
    model = torch.load(str(pa).strip())

    # 先将模型的调整为half()格式，再将其贴到GPU
    model = model.get("model").half().to(check_GPU())

    # 假设模型期望的输入是3个通道的图像
    input_tensor = torch.randn(1, 3, 640, 640)

    # 同理，将数据转化成半精度之后再加载到GPU
    input_tensor = input_tensor.half().to(check_GPU())

    # 调用profile函数计算FLOPs和参数量
    macs, params = profile(model, inputs=(input_tensor,))

    # 将macs转化成flops
    flops = macs / 1E9 * 2
    params = params / 1E6

    print('flops:{}'.format(flops))
    print('params:{}'.format(params))


def predict():
    # Load a model
    # model = YOLO('yolov8n.pt')  # 加载官方的模型权重作评估
    model = YOLO("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/RFB+ASFF/weights/best.pt")  # 加载自定义的模型权重作评估
    # metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    metrics = model.val(data="/home/youtian/Documents/pro/pyCode/easy_YOLOv8/ultralytics/cfg/datasets/HSTS6.yaml")
    # 在一个新的数据集上做评估，传绝对路径
    print(metrics.box.map)  # map50-95
    print(metrics.box.map50)  # map50
    print(metrics.box.map75)  # map75
    print(metrics.box.maps)  # 包含每个类别的map50-95列表


if __name__ == "__main__":
    # Para4pt(model)
    # FLOPs_Para4pt()

    # loss_compara_pic("./loss_csv")
    # calc_instance()
    train()
    # test_video()
    # test_folders()
    # test_img()
    # tracker()
    # onnx()
    # predict()
