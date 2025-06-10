from typing import Union, List, Optional, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import shutil
import sys
import os
import torch
from pathlib import Path
from typing import Union
import cv2
import netron
import time
from ultralytics import YOLO

os.environ["WANDB_MODE"] = "offline"  # 离线模式


def train():
    # Load a model
    model = YOLO('yolov8n.yaml')
    # model = RTDETR('rtdetr-l.yaml')
    # print(model)
    # model = YOLO('yolov8m.pt')

    # 做预训练
    # model = YOLO('yolov8x.pt')
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')

    # Train the model
    # model.train(data="HSTS6.yaml", epochs=150, imgsz=640)
    model.train(data="coco128.yaml", epochs=5, imgsz=640)


def train_lowght():
    # Load a model
    model = YOLO('yolov8l.yaml')
    model.train(data="tielu.yaml", epochs=5, imgsz=640)


def onnx(path: Union[str, Path] = "/home/youtian/Documents/pro/pyCode/Dedark-YOLO/epoch20.pt"):
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
    model = YOLO("/home/youtian/Documents/pro/pyCode/Dedark-YOLO/runs/detect/6-1-1833/no-re_loss.pt")
    img = cv2.imread(
        "/home/youtian/Documents/pro/pyCode/datasets/darkpic_test/192.168.39.20_20240727_060304_2246886_2.jpg")
    # img = cv2.imread("/home/youtian/Documents/pro/pyCode/datasets/tielu-yolo/images/test/192.168.39.20_20240727_060304_2246886_2.jpg")
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




def test_folders(
        model_path: Union[
            str, Path] = "/home/youtian/best.pt",
        srcpath: Union[str, Path] = "/home/youtian/val",
        method: str = "YOLO",
        output_size: Tuple[int, int] = (1080, 720),
        confidence_threshold: float = 0.25,
        save_txt: bool = False,
        save_json: bool = True,
        supported_formats: List[str] = None
) -> dict:
    """
    优化的批量图片检测函数

    Args:
        model_path: 模型权重文件路径
        srcpath: 源图片文件夹路径
        method: 检测方法 ("YOLO" 或 "RTDETR")
        output_size: 输出图片尺寸 (width, height)
        confidence_threshold: 置信度阈值
        save_txt: 是否保存txt格式的检测结果
        save_json: 是否保存json格式的检测结果
        supported_formats: 支持的图片格式列表

    Returns:
        dict: 包含检测统计信息的字典
    """

    # 默认支持的图片格式
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # 路径处理
    model_path = Path(model_path)
    src_path = Path(srcpath)

    # 验证路径
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not src_path.exists():
        raise FileNotFoundError(f"源文件夹不存在: {src_path}")

    # 加载模型
    print(f"正在加载模型: {model_path}")
    try:
        if method.lower() == "rtdetr":
            # model = RTDETR(str(model_path))  # 如果需要支持RTDETR
            raise NotImplementedError("RTDETR support not implemented yet")
        else:
            model = YOLO(model_path)
            model.conf = confidence_threshold  # 设置置信度阈值
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_path.parts[-3] if len(model_path.parts) >= 3 else model_path.stem
    dst_folder = Path(sys.path[0]) / f"{model_name}_test_results_{timestamp}"

    # 清理并创建输出文件夹
    if dst_folder.exists():
        shutil.rmtree(dst_folder)
    dst_folder.mkdir(parents=True)

    # 创建子文件夹
    images_folder = dst_folder / "images"
    images_folder.mkdir()

    if save_txt:
        labels_folder = dst_folder / "labels"
        labels_folder.mkdir()

    # 获取所有图片文件
    image_files = []
    for fmt in supported_formats:
        image_files.extend(src_path.glob(f"*{fmt}"))
        image_files.extend(src_path.glob(f"*{fmt.upper()}"))

    if not image_files:
        raise ValueError(f"在 {src_path} 中未找到支持的图片文件")

    print(f"找到 {len(image_files)} 张图片")

    # 初始化统计变量
    total_time = 0
    successful_detections = 0
    failed_detections = 0
    detection_results = []

    # 批量处理图片
    with tqdm(image_files, desc="处理图片", unit="张") as pbar:
        for img_path in pbar:
            try:
                # 记录开始时间
                start_time = time.time()

                # 读取图片
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"警告: 无法读取图片 {img_path}")
                    failed_detections += 1
                    continue

                # 进行推理
                results = model(img, verbose=False)  # verbose=False 减少输出

                # 记录结束时间
                end_time = time.time()
                inference_time = end_time - start_time
                total_time += inference_time

                # 处理检测结果
                result = results[0]
                annotated_img = result.plot()

                # 调整图片尺寸
                resized_img = cv2.resize(annotated_img, output_size)

                # 保存标注图片
                output_img_path = images_folder / img_path.name
                cv2.imwrite(str(output_img_path), resized_img)

                # 保存检测结果 (txt格式)
                if save_txt and hasattr(result, 'boxes') and result.boxes is not None:
                    txt_path = labels_folder / f"{img_path.stem}.txt"
                    save_detection_txt(result, txt_path, img.shape)

                # 收集统计信息
                detection_info = {
                    'filename': img_path.name,
                    'inference_time': inference_time,
                    'detection_count': len(result.boxes) if result.boxes is not None else 0,
                    'image_size': img.shape[:2],
                    'confidences': result.boxes.conf.tolist() if result.boxes is not None else []
                }
                detection_results.append(detection_info)

                successful_detections += 1

                # 更新进度条信息
                avg_time = total_time / successful_detections if successful_detections > 0 else 0
                pbar.set_postfix({
                    'avg_time': f"{avg_time:.3f}s",
                    'detections': len(result.boxes) if result.boxes is not None else 0
                })

            except Exception as e:
                print(f"处理图片 {img_path} 时出错: {e}")
                failed_detections += 1
                continue

    # 计算统计信息
    avg_inference_time = total_time / successful_detections if successful_detections > 0 else 0
    total_detections = sum(info['detection_count'] for info in detection_results)

    # 统计结果
    stats = {
        'model_path': str(model_path),
        'source_path': str(src_path),
        'output_path': str(dst_folder),
        'timestamp': timestamp,
        'total_images': len(image_files),
        'successful_detections': successful_detections,
        'failed_detections': failed_detections,
        'total_inference_time': total_time,
        'average_inference_time': avg_inference_time,
        'total_objects_detected': total_detections,
        'confidence_threshold': confidence_threshold,
        'output_size': output_size,
        'detection_details': detection_results
    }

    # 保存统计结果
    if save_json:
        stats_path = dst_folder / "detection_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印结果摘要
    print_detection_summary(stats)

    return stats


def save_detection_txt(result, txt_path: Path, img_shape: tuple):
    """保存检测结果到txt文件 (YOLO格式)"""
    try:
        with open(txt_path, 'w') as f:
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    x, y, w, h = boxes.xywhn[i].tolist()  # 归一化坐标
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
    except Exception as e:
        print(f"保存txt文件失败 {txt_path}: {e}")


def print_detection_summary(stats: dict):
    """打印检测结果摘要"""
    print("\n" + "=" * 60)
    print("检测结果摘要")
    print("=" * 60)
    print(f"📁 输出文件夹: {stats['output_path']}")
    print(f"📊 处理统计:")
    print(f"   • 总图片数: {stats['total_images']}")
    print(f"   • 成功处理: {stats['successful_detections']}")
    print(f"   • 处理失败: {stats['failed_detections']}")
    print(f"   • 成功率: {stats['successful_detections'] / stats['total_images'] * 100:.1f}%")

    print(f"⏱️  性能统计:")
    print(f"   • 总推理时间: {stats['total_inference_time']:.2f}s")
    print(f"   • 平均推理时间: {stats['average_inference_time']:.3f}s")
    print(f"   • 推理速度: {1 / stats['average_inference_time']:.1f} FPS"
          if stats['average_inference_time'] > 0 else "   • 推理速度: N/A")

    print(f"🎯 检测统计:")
    print(f"   • 总检测目标数: {stats['total_objects_detected']}")
    print(f"   • 平均每张图片目标数: {stats['total_objects_detected'] / stats['successful_detections']:.1f}"
          if stats['successful_detections'] > 0 else "   • 平均每张图片目标数: 0")

    # 置信度统计
    if stats['detection_details']:
        all_confidences = []
        for detail in stats['detection_details']:
            all_confidences.extend(detail['confidences'])

        if all_confidences:
            print(f"📈 置信度统计:")
            print(f"   • 最高置信度: {max(all_confidences):.3f}")
            print(f"   • 最低置信度: {min(all_confidences):.3f}")
            print(f"   • 平均置信度: {sum(all_confidences) / len(all_confidences):.3f}")

    print("=" * 60)



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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pa = Path("/home/youtian/Documents/pro/pyCode/ultralytics-YOLOv8/yolov8l.pt")
    model = torch.load(str(pa).strip())

    # 先将模型的调整为half()格式，再将其贴到GPU
    model = model.get("model").half().to(device)

    # 假设模型期望的输入是3个通道的图像
    input_tensor = torch.randn(1, 3, 640, 640)

    # 同理，将数据转化成半精度之后再加载到GPU
    input_tensor = input_tensor.half().to(device)

    # 调用profile函数计算FLOPs和参数量
    macs, params = profile(model, inputs=(input_tensor,))

    # 将macs转化成flops
    flops = macs / 1E9 * 2
    params = params / 1E6

    print('flops:{}'.format(flops))
    print('params:{}'.format(params))


def calculate_detection_metrics(metrics, class_names=None):
    """
    计算检出率和漏检率指标
    检出率（Detection Rate, DR）= TP / (TP + FN) = 1 - 漏检率
    漏检率（False Negative Rate, FNR）= FN / (FN + TP) = FN / Total_Ground_Truth

    Args:
        metrics: YOLO validation metrics
        class_names: 类别名称列表，如 ['person', 'debrisflow', 'rockfall']

    Returns:
        dict: 包含总体和各类别检出率、漏检率的字典
    """
    # 获取混淆矩阵
    confusion_matrix = metrics.confusion_matrix

    if confusion_matrix is None:
        print("Warning: No confusion matrix available")
        return None

    # 获取混淆矩阵数据
    matrix = confusion_matrix.matrix
    nc = confusion_matrix.nc  # 类别数量

    # 计算每个类别的TP, FN, DR, FNR
    tp_per_class = []
    fn_per_class = []
    dr_per_class = []  # Detection Rate
    fnr_per_class = []  # False Negative Rate

    for i in range(nc):
        # TP: 对角线元素
        tp = matrix[i, i]
        # FN: 该类别列的总和减去TP (真实标签为该类但被分类为其他类或未检出)
        fn = matrix[:, i].sum() - tp
        # 计算检出率和漏检率
        total_gt = tp + fn  # 该类别的真实目标总数
        dr = tp / total_gt if total_gt > 0 else 0.0  # 检出率 = TP / (TP + FN)
        fnr = fn / total_gt if total_gt > 0 else 0.0  # 漏检率 = FN / (TP + FN)

        tp_per_class.append(tp)
        fn_per_class.append(fn)
        dr_per_class.append(dr)
        fnr_per_class.append(fnr)

    # 计算总体指标
    total_tp = sum(tp_per_class)
    total_fn = sum(fn_per_class)
    total_gt = total_tp + total_fn
    overall_dr = total_tp / total_gt if total_gt > 0 else 0.0  # 总体检出率
    overall_fnr = total_fn / total_gt if total_gt > 0 else 0.0  # 总体漏检率

    # 准备结果
    results = {
        'overall_detection_rate': overall_dr,
        'overall_miss_rate': overall_fnr,
        'total_tp': total_tp,
        'total_fn': total_fn,
        'total_ground_truth': total_gt,
        'class_detection_rates': {},
        'class_miss_rates': {},
        'class_details': {}
    }

    # 添加各类别详细信息
    for i in range(nc):
        class_name = class_names[i] if class_names and i < len(class_names) else f'class_{i}'
        results['class_detection_rates'][class_name] = dr_per_class[i]
        results['class_miss_rates'][class_name] = fnr_per_class[i]
        results['class_details'][class_name] = {
            'tp': tp_per_class[i],
            'fn': fn_per_class[i],
            'ground_truth': tp_per_class[i] + fn_per_class[i],
            'detection_rate': dr_per_class[i],
            'miss_rate': fnr_per_class[i]
        }

    return results


def print_detection_metrics_report(detection_results):
    """
    打印检出率和漏检率综合报告

    Args:
        detection_results: calculate_detection_metrics函数的返回结果
    """
    if detection_results is None:
        print("No detection metrics data available")
        return

    print("\n" + "=" * 70)
    print("检出率与漏检率分析报告 (Detection & Miss Rate Analysis Report)")
    print("=" * 70)

    # 总体指标
    dr_pct = detection_results['overall_detection_rate'] * 100
    mr_pct = detection_results['overall_miss_rate'] * 100

    print(f"\n📊 总体指标:")
    print(f"   🎯 总体检出率: {detection_results['overall_detection_rate']:.4f} ({dr_pct:.2f}%)")
    print(f"   🚨 总体漏检率: {detection_results['overall_miss_rate']:.4f} ({mr_pct:.2f}%)")
    print(f"   📈 总TP: {detection_results['total_tp']}")
    print(f"   📉 总FN: {detection_results['total_fn']}")
    print(f"   📋 总GT: {detection_results['total_ground_truth']}")

    # 各类别详情
    print(f"\n📋 各类别详细指标:")
    print("-" * 70)
    print(f"{'类别':<12} {'检出率':<10} {'漏检率':<10} {'TP':<6} {'FN':<6} {'GT总数':<8}")
    print("-" * 70)

    for class_name, details in detection_results['class_details'].items():
        dr_pct = details['detection_rate'] * 100
        mr_pct = details['miss_rate'] * 100
        print(
            f"{class_name:<12} {dr_pct:<9.2f}% {mr_pct:<9.2f}% {details['tp']:<6} {details['fn']:<6} {details['ground_truth']:<8}")

    print("-" * 70)

    # 性能分析
    print(f"\n💡 性能分析:")
    overall_dr = detection_results['overall_detection_rate']
    overall_mr = detection_results['overall_miss_rate']

    if overall_dr > 0.9:
        print("   ✅ 优秀: 检出率>90%，模型检测能力强")
    elif overall_dr > 0.8:
        print("   ⚡ 良好: 检出率80-90%，性能较好")
    elif overall_dr > 0.7:
        print("   📈 中等: 检出率70-80%，有改进空间")
    else:
        print("   ⚠️  较差: 检出率<70%，需要重点优化")

    if overall_mr > 0.3:
        print("   🚨 高漏检: 漏检率>30%，严重影响实用性")
    elif overall_mr > 0.2:
        print("   ⚠️  中等漏检: 漏检率20-30%，需要关注")
    elif overall_mr > 0.1:
        print("   📊 低漏检: 漏检率10-20%，可接受范围")
    else:
        print("   🎯 极低漏检: 漏检率<10%，表现优秀")

    # 找出问题类别
    worst_dr_class = min(detection_results['class_details'].items(),
                         key=lambda x: x[1]['detection_rate'])
    worst_mr_class = max(detection_results['class_details'].items(),
                         key=lambda x: x[1]['miss_rate'])

    print(f"\n🎯 重点关注:")
    if worst_dr_class[1]['detection_rate'] < 0.8:
        print(f"   📉 '{worst_dr_class[0]}'类别检出率最低({worst_dr_class[1]['detection_rate'] * 100:.1f}%)")
    if worst_mr_class[1]['miss_rate'] > 0.2:
        print(f"   📈 '{worst_mr_class[0]}'类别漏检率最高({worst_mr_class[1]['miss_rate'] * 100:.1f}%)")

    # 改进建议
    print(f"\n🔧 改进建议:")
    if overall_dr < 0.8 or overall_mr > 0.2:
        print("   - 降低置信度阈值以提高检出率")
        print("   - 增加困难样本和边界案例的训练数据")
        print("   - 检查数据标注质量和一致性")
        print("   - 考虑调整网络结构或损失函数")
        print("   - 进行数据增强以提高模型鲁棒性")
    else:
        print("   ✅ 模型检出性能已达到良好水平")


def predict():
    # Load a model
    # model = YOLO('yolov8n.pt')  # 加载官方的模型权重作评估
    model = YOLO("/home/youtian/Documents/pro/pyCode/Dedark-YOLO/runs/detect/DedarkDet/weights/best.pt")  # 加载自定义的模型权重作评估

    # 定义你的类别名称
    class_names = ['person', 'debrisflow', 'rockfall']

    # 进行模型验证
    # metrics = model.val()  # 不需要传参，这里定义的模型会自动在训练的数据集上作评估
    metrics = model.val(data="/home/youtian/Documents/pro/pyCode/Dedark-YOLO/ultralytics/cfg/datasets/tielu.yaml")
    # 在一个新的数据集上做评估，传绝对路径

    # 原有的指标
    print("=" * 50)
    print("传统检测指标:")
    print("=" * 50)
    print(f"mAP50-95: {metrics.box.map:.4f}")  # map50-95
    print(f"mAP50: {metrics.box.map50:.4f}")  # map50

    print(f"AP75 per class: person-{metrics.box.map75[0]:.4f},"
          f" debrisflow-{metrics.box.map75[1]:.4f},"
          f" rockfall-{metrics.box.map75[2]:.4f}"
          f" mAP75:{sum(metrics.box.map75) / len(metrics.box.map75)}")  # map75

    print(f"F1 scores: person-{metrics.box.f1s[0]:.4f},"
          f" debrisflow-{metrics.box.f1s[1]:.4f},"
          f" rockfall-{metrics.box.f1s[2]:.4f}")  # f1 score

    print(f"mf1: {metrics.box.mf1:.4f}")

    # 新增的检出率和漏检率分析
    detection_results = calculate_detection_metrics(metrics, class_names)
    print_detection_metrics_report(detection_results)

    return metrics, detection_results


if __name__ == "__main__":
    # train_lowght()
    # test_img()

#############---predict---##############

    # 使用原有的predict函数（包含检出率和漏检率）
    # predict()

    # 或者使用带有详细分析的版本
    # predict_with_detailed_analysis()

    # onnx()

#############---test—folder---##############
    try:
        # 基础使用
        stats = test_folders(
            model_path="/home/youtian/Documents/pro/pyCode/Dedark-YOLO/runs/detect/DedarkDet/weights/best.pt",
            srcpath="/home/youtian/Documents/pro/pyCode/Dedark-YOLO/exp/test",
            confidence_threshold=0.3,
            save_txt=True,
            save_json=True
        )

    except Exception as e:
        print(f"错误: {e}")