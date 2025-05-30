import os
import shutil
import xml.etree.ElementTree as ET
from PIL import Image
import yaml


"""
    convert voc format to yolo format

VOC_dataset/
├── Annotations/  # 包含 XML 标注文件
├── JPEGImages/   # 包含图像文件
└── ImageSets/Main/  # 包含 train.txt, val.txt 等划分文件
"""

"""
YOLO_dataset/
├── images/  # 存放图像
│   ├── train/
│   ├── test/
├── labels/  # 存放 YOLO 格式的标注文件
│   ├── train/
│   ├── test/
└── data.yaml  # YOLO 配置文件
"""


# 配置路径和类别
VOC_DIR = '/home/youtian/Documents/pro/pyCode/datasets/tielu1386 -yolo/TLdevkit/TL2025'  # Pascal VOC 数据集根目录
YOLO_DIR = '/home/youtian/Documents/pro/pyCode/datasets/tielu-yolo'  # YOLO 数据集输出目录
CLASSES = ['person']  # 替换为你的类别列表
CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}


def create_yolo_dirs():
    """创建 YOLO 数据集目录结构"""
    os.makedirs(f"{YOLO_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/images/test", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/test", exist_ok=True)


def copy_images(split='train'):
    """复制图像到 YOLO 数据集的对应目录"""
    src_dir = f"{VOC_DIR}/JPEGImages"
    dst_dir = f"{YOLO_DIR}/images/{split}"
    split_file = f"{VOC_DIR}/ImageSets/Main/{split}.txt"

    if not os.path.exists(split_file):
        print(f"Warning: {split_file} does not exist!")
        return

    with open(split_file, 'r') as f:
        for line in f:
            img_name = line.strip() + '.jpg'  # 假设图像为 .jpg 格式
            src_path = os.path.join(src_dir, img_name)
            dst_path = os.path.join(dst_dir, img_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Image {src_path} not found!")


def convert_voc_to_yolo(split='train'):
    """将 VOC 的 XML 标注转换为 YOLO 格式"""
    xml_dir = f"{VOC_DIR}/Annotations"
    img_dir = f"{VOC_DIR}/JPEGImages"
    out_label_dir = f"{YOLO_DIR}/labels/{split}"

    split_file = f"{VOC_DIR}/ImageSets/Main/{split}.txt"
    if not os.path.exists(split_file):
        print(f"Warning: {split_file} does not exist!")
        return

    with open(split_file, 'r') as f:
        for line in f:
            xml_name = line.strip() + '.xml'
            xml_path = os.path.join(xml_dir, xml_name)

            if not os.path.exists(xml_path):
                print(f"Warning: XML {xml_path} not found!")
                continue

            # 解析 XML 文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像尺寸
            img_name = root.find('filename').text
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"Warning: Image {img_path} not found!")
                continue
            img = Image.open(img_path)
            img_width, img_height = img.size

            # 输出 YOLO 标签文件
            txt_name = xml_name.replace('.xml', '.txt')
            txt_path = os.path.join(out_label_dir, txt_name)

            with open(txt_path, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in CLASS_TO_ID:
                        print(f"Warning: Class {class_name} not in class list!")
                        continue
                    class_id = CLASS_TO_ID[class_name]

                    # 获取边界框
                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # 转换为 YOLO 格式（归一化）
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    # 写入 YOLO 格式
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def create_yaml_file():
    """创建 YOLO 的 data.yaml 配置文件"""
    data = {
        'train': f'./{YOLO_DIR}/images/train',
        'val': f'./{YOLO_DIR}/images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    with open(f"{YOLO_DIR}/data.yaml", 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def main():
    # 创建目录结构
    create_yolo_dirs()

    # 复制训练和验证集图像
    copy_images('train')
    copy_images('test')

    # 转换训练和验证集标注
    convert_voc_to_yolo('train')
    convert_voc_to_yolo('test')

    # 创建 data.yaml 文件
    create_yaml_file()

    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()