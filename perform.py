import shutil
import sys
import os
import torch
from pathlib import Path
from typing import Union
import cv2
import netron
import time
import numpy as np
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
        model_path: str = "/home/youtian/Documents/pro/pyCode/easy_YOLOv8/runs/detect/YOLOv8l+CHST6+32/weights/best.pt",
        srcpath: str = "/home/youtian/Documents/pro/pyCode/datasets/HSTS6/CHTS6/images/val",
        mothed: str = "YOLO") -> None:
    # 加载权重model
    if not isinstance(model_path, Path):
        model_path = Path(model_path)
    # if mothed.lower() == "rtdetr":
    #     model = RTDETR(str(model_path))
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
    model = YOLO("/home/youtian/Documents/pro/pyCode/Dedark-YOLO/runs/detect/baseline/weights/best.pt")  # 加载自定义的模型权重作评估

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
    print(f"mAP75: {metrics.box.map75:.4f}")  # map75
    print(f"F1 scores: {metrics.box.f1s}")  # f1 score
    print(f"mAPs per class: {metrics.box.maps}")  # 包含每个类别的map50-95列表

    # 新增的检出率和漏检率分析
    detection_results = calculate_detection_metrics(metrics, class_names)
    print_detection_metrics_report(detection_results)

    return metrics, detection_results


def predict_with_detailed_analysis():
    """
    带有详细分析的预测函数，包括漏检率和其他指标的综合分析
    """
    # Load a model
    model = YOLO("/home/youtian/Documents/pro/pyCode/Dedark-YOLO/runs/detect/baseline/weights/best.pt")

    # 定义你的类别名称
    class_names = ['person', 'debrisflow', 'rockfall']

    # 进行模型验证
    metrics = model.val(data="/home/youtian/Documents/pro/pyCode/Dedark-YOLO/ultralytics/cfg/datasets/tielu.yaml")

    # 计算检出率和漏检率
    detection_results = calculate_detection_metrics(metrics, class_names)

    # 综合报告
    print("\n" + "=" * 70)
    print("模型性能综合分析报告")
    print("=" * 70)

    # 基础指标
    print(f"\n🎯 检测精度指标:")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"   mAP@0.75:    {metrics.box.map75:.4f}")
    print(f"   平均精确率:   {metrics.box.mp:.4f}")
    print(f"   平均召回率:   {metrics.box.mr:.4f}")

    # 检出率和漏检率指标
    if detection_results:
        print(f"\n🎯 检出率指标:")
        print(
            f"   总体检出率: {detection_results['overall_detection_rate']:.4f} ({detection_results['overall_detection_rate'] * 100:.2f}%)")
        print(f"\n🚨 漏检率指标:")
        print(
            f"   总体漏检率: {detection_results['overall_miss_rate']:.4f} ({detection_results['overall_miss_rate'] * 100:.2f}%)")

        # 各类别对比
        print(f"\n📊 各类别性能对比:")
        print(f"{'类别':<12} {'mAP@0.5':<10} {'召回率':<10} {'检出率':<10} {'漏检率':<10} {'F1':<10}")
        print("-" * 70)

        for i, class_name in enumerate(class_names):
            if i < len(metrics.box.maps):
                map50 = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') and i < len(metrics.box.ap50) else 0
                recall = metrics.box.r[i] if hasattr(metrics.box, 'r') and i < len(metrics.box.r) else 0
                detection_rate = detection_results['class_detection_rates'].get(class_name, 0)
                miss_rate = detection_results['class_miss_rates'].get(class_name, 0)
                f1 = metrics.box.f1[i] if hasattr(metrics.box, 'f1') and i < len(metrics.box.f1) else 0

                print(
                    f"{class_name:<12} {map50:<10.3f} {recall:<10.3f} {detection_rate * 100:<9.1f}% {miss_rate * 100:<9.1f}% {f1:<10.3f}")

    # 性能评估
    print(f"\n💡 模型评估:")
    overall_map = metrics.box.map
    overall_detection_rate = detection_results['overall_detection_rate'] if detection_results else 0
    overall_miss_rate = detection_results['overall_miss_rate'] if detection_results else 0

    if overall_map > 0.7 and overall_detection_rate > 0.9:
        print("   ✅ 优秀: 高精度高检出率，模型性能优秀")
    elif overall_map > 0.5 and overall_detection_rate > 0.8:
        print("   ⚡ 良好: 精度和检出率都在良好范围")
    elif overall_miss_rate > 0.3:
        print("   ⚠️  注意: 漏检率较高，可能影响实际应用")
    else:
        print("   📈 需改进: 建议进一步优化模型")

    return metrics, detection_results


if __name__ == "__main__":
    # train_lowght()
    # test_img()

    # 使用原有的predict函数（包含检出率和漏检率）
    predict()

    # 或者使用带有详细分析的版本
    # predict_with_detailed_analysis()

    # onnx()