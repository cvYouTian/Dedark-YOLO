# Dedark-YOLO：低光照增强的 YOLOv8

Dedark-YOLO 是 YOLOv8 的增强版本，旨在提升低光照条件下的目标检测性能。本项目在 YOLOv8 架构中集成了低光照图像增强模块（`lowlight_recovery`），引入了自定义损失函数（`RcoveryDetectionLoss`）以融合低光照恢复损失，并支持将 Pascal VOC 数据集转换为 YOLO 格式。此外，项目还对模型配置、验证和可视化进行了修改，以优化低光照检测。

## 功能

- **低光照增强**：集成了 `lowlight_recovery` 模块，在检测前增强低光照图像-。
- **自定义损失函数**：在 `RcoveryDetectionLoss` 中将 `recovery_loss` 添加到分类损失（`cls_loss`），保持 `loss_items` 形状为 `[3]`。
- **数据转换**：支持将 Pascal VOC 的 XML 标注文件转换为 YOLO 的 `.txt` 格式。
- **低光照图像生成**：包含 `lowlight_maker` 用于生成低光照图像，`lowlight_process` 用于保存处理后的图像。
- **模型配置**：修改 `yolov8.yaml`，支持 `lowlight_recovery` 模块并调整通道配置。
- **验证优化**：将 YOLOv8 的半精度（half-precision）改为 `float32`，提升稳定性。
- **自定义 PR 曲线**：修改 `metrics` 中的 `plot_pr_curve`，便于对比实验的可视化。
- **低光照强度调节**：通过 `lowlight_recovery` 中的 `lowlight_param` 参数调节低光照强度。

## 安装

### 环境要求
- Python 3.8 或更高版本
- PyTorch 1.7 或更高版本（推荐使用支持 CUDA 的 GPU 版本）
- OpenCV（`opencv-python`）
- NumPy
- Ultralytics YOLOv8（`ultralytics` 包）
- **注意**：环境中不得包含 TensorFlow，否则可能引发冲突。如已安装，请删除 TensorFlow 或重新构建环境。

### 安装步骤
1. 克隆代码仓库：
   ```bash
   git clone https://github.com/your-username/Dedark-YOLO.git
   cd Dedark-YOLO
   ```

2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. 安装依赖：
   ```bash
   pip install torch torchvision opencv-python numpy ultralytics
   ```

4. 下载 YOLOv8n 预训练权重：
   - 从 [Ultralytics YOLOv8 发布页面](https://github.com/ultralytics/ultralytics/releases) 下载 `yolov8n.pt`，或运行：
     ```bash
     wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
     ```
   - 将 `yolov8n.pt` 放置在项目根目录。

## 数据准备

### Pascal VOC 转 YOLO 格式
项目支持将 Pascal VOC 数据集转换为 YOLO 格式。

1. **数据集结构**：
   - Pascal VOC：
     ```plaintext
     VOCdevkit/
     ├── VOC2007/
     │   ├── Annotations/      # XML 标注文件
     │   ├── JPEGImages/       # 图像文件
     │   ├── ImageSets/Main/   # train.txt, val.txt
     ```
   - YOLO（输出）：
     ```plaintext
     dataset/
     ├── images/
     │   ├── train/
     │   └── val/
     ├── labels/
     │   ├── train/
     │   └── val/
     ├── data.yaml
     ```

2. **运行转换脚本**：
   - 使用提供的 `convert_voc_to_yolo.py` 将 VOC 的 XML 标注转换为 YOLO 的 `.txt` 格式：
     ```bash
     python convert_voc_to_yolo.py
     ```
   - 修改脚本中的 `VOC_ROOT` 和 `YOLO_ROOT` 为你的数据集路径。
   - 输出：生成 `dataset/` 目录，包含图像、标签和 `data.yaml`。

3. **生成低光照图像**：
   - 使用 `lowlight_maker.py` 生成低光照版本的图像：
     ```bash
     python lowlight_maker.py --input dataset/images/train --output dataset/lowlight_images/train
     ```
   - 在 `ultralytics/nn/modules/llie.py` 中调整 `lowlight_param` 参数（默认：`random.uniform(5, 10)`）以控制低光照强度。

## 使用方法

### 训练模型
使用转换后的 YOLO 格式数据集进行训练：

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # 加载预训练权重
model.train(
    data='dataset/data.yaml',  # data.yaml 路径
    epochs=5,
    imgsz=640,
    batch=4,
    workers=0,
    drop_last=True,
    device=0  # 使用 GPU
)
```

### 验证模型
在验证集上评估模型性能：

```python
model.val(data='dataset/data.yaml', batch=4)
```

### 保存低光照增强图像
- `lowlight_recovery` 模块会自动将增强后的图像保存到 `filtered_images/` 目录。
- 在 `ultralytics/nn/modules/llie.py` 中，`save_filtered_image` 方法生成文件名如 `filtered_{batch_idx}_{image_idx}.jpg`。
- 调整保存频率（例如每 10 个 batch 保存一次）以减少磁盘占用：
  ```python
  if batch_idx % 10 == 0:
      self.save_filtered_image(filtered_image, batch_idx)
  ```

### 调节低光照强度
- 修改 `ultralytics/nn/modules/llie.py` 中的 `lowlight_param`：
  ```python
  self.lowlight_param = random.uniform(5, 10)  # 调整范围，例如 (3, 8)
  ```
- 较小的值（如 3）生成较轻的低光照效果，较大的值（如 10）生成更暗的图像。

## 项目结构

- **`data/`**：
  - `lowlight_maker.py`：生成低光照图像。
  - `convert_voc_to_yolo.py`：将 Pascal VOC XML 转换为 YOLO `.txt`。
- **`ultralytics/`**：
  - `nn/modules/llie.py`：实现 `lowlight_recovery` 模块。
  - `nn/tasks.py`：修改 `DetectionModel` 以支持低光照增强。
  - `utils/loss.py`：实现 `RcoveryDetectionLoss`，融合 `recovery_loss`。
  - `utils/lowlight_process.py`：保存低光照处理图像。
  - `engine/validator.py`：将半精度改为 `float32`。
  - `utils/metrics.py`：修改 PR 曲线绘制。
- **`yolov8.yaml`**：模型配置文件，注意仅兼容 YOLOv8-L 版本（ASFF 和 RBF 模块）。
- **`default.yaml`**：添加 `lrl` 参数，控制低光照损失权重。
- **`perform.py`**：训练和测试脚本。

## 注意事项

1. **模型兼容性**：
   - `yolov8.yaml` 中的 ASFF 和 RBF 模块仅适用于 YOLOv8-L 版本，可能不兼容其他版本（如 YOLOv8n）。请检查配置或使用 YOLOv8-L。
   - 如果遇到通道不匹配错误（例如 `RuntimeError: expected input to have 256 channels, but got 64 channels`），检查 `yolov8.yaml` 中 `lowlight_recovery` 的输出通道与后续层的输入通道是否一致。

2. **数据集准备**：
   - 确保 Pascal VOC 数据集的 `train.txt` 和 `val.txt` 存在。
   - 验证转换后的 YOLO 标签文件（`.txt`）格式是否正确（归一化坐标）。

3. **环境冲突**：
   - 移除 TensorFlow 依赖以避免冲突：
     ```bash
     pip uninstall tensorflow
     ```

4. **预训练权重**：
   - 运行前必须下载 `yolov8n.pt`，否则会报错。

5. **调试日志**：
   - 检查 `llie.py` 和 `tasks.py` 的形状日志，确保 `lowlight_recovery` 输出形状正确。
   - 监控 `loss.py` 中的 `recovery_loss`，调整 `lrl`（默认在 `default.yaml`）以平衡检测和增强损失。

## 贡献

欢迎提交 Issue 或 Pull Request 来改进 Dedark-YOLO！请确保代码遵循项目风格，并提供详细的测试结果。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。请参阅 `LICENSE` 文件了解详情。

## 联系

如有问题，请通过 [GitHub Issues](https://github.com/your-username/Dedark-YOLO/issues) 联系，或发送邮件至 tianyou9890@163.com。