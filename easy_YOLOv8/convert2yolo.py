import os
from PIL import Image


def convert_to_yolo_format(label_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 分割图像路径和边界框
        parts = line.split()
        image_path = parts[0]  # 第一个元素是图像路径
        boxes = parts[1:]  # 剩余元素是边界框

        try:
            # 读取图像尺寸
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"无法读取图像 {image_path}: {e}")
            continue

        yolo_labels = []
        for box in boxes:
            # 分割坐标和类别
            box_data = box.split(',')
            if len(box_data) != 5:
                print(f"无效的边界框格式: {box} (图像: {image_path})")
                continue

            try:
                # 解析坐标和类别
                x_min = int(box_data[0])
                y_min = int(box_data[1])
                x_max = int(box_data[2])
                y_max = int(box_data[3])
                class_id = int(box_data[4])
            except (ValueError, IndexError):
                print(f"坐标解析失败: {box} (图像: {image_path})")
                continue

            # 计算归一化坐标
            x_center = (x_min + x_max) / (2 * width)
            y_center = (y_min + y_max) / (2 * height)
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height

            # 验证坐标范围
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                print(f"坐标超出范围 [0,1]: {box} (图像: {image_path})")
                continue

            yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # 生成标签文件
        if yolo_labels:
            base_name = os.path.basename(image_path)
            file_name, _ = os.path.splitext(base_name)
            txt_name = f"{file_name}.txt"
            txt_path = os.path.join(output_dir, txt_name)

            with open(txt_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

# 使用示例
label_file = '/home/youtian/Documents/pro/pyCode/Dedark-YOLO/pytorch_file/data/tielu_train.txt'  # 替换为您的标签文件路径
output_dir = '/home/youtian/Documents/pro/pyCode/Dedark-YOLO/pytorch_file/data'     # 替换为输出目录路径
convert_to_yolo_format(label_file, output_dir)