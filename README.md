# data
- lowlight_maker(将图片转化成低光照的图片)
- 将xml的标注文件转化成txt的

# validator
- 将yolo8的半精度改成float32
 
# yolov8.yaml
- asff、rbf只使用于l版本，这里的参数的调整没有写好，导致只兼容yolo的l版本

# 在default.yaml中添加 lrl用于低光照损失的权值

# 为了实现低光照增强，我更改了detectionmodel

# 在utils中加入lowlight_process函数用于保存加暗处理的图像

# 如果要调节暗处理的强度
- 调节lite的lowlight的参数