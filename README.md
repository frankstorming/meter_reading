# meter_reading
A traditional way to solve pointer meter reading
## image文件夹
58张测试图片和11张模板图片，均来源于实际拍照截图
## result文件夹
对58张测试图片指针的标注，在一红一黑双指针图片中，白色线表示红指针，绿色线表示黑指针
## meter_reading.py
采用模板匹配 + 直线拟合 + 表盘读数的方法读取仪表中指针所指刻度

## 提升准确率可采用的方法

### 一、传统方法
1、选用更规范的模板图片
2、选用其他模板匹配方法
3、调整模板图尺寸，
4、直线拟合时采用更高精度(实验中采用1度，选用0.5度可达更佳效果)
5、直线拟合选取更优直线宽度(实验中采用thickness=2，可调整为其他)
6、高斯去噪等滤波手段

### 二、目标检测方法/关键点检测模型
1、YOLOX等目标检测方法识别表盘 2、识别数字、指针、指针旋转原点 3、欧式距离求相邻数字 4、字符识别模型识别数字 5、根据相邻数字求得指针所指数值

## 适用场景
1、同类别图片相差不大，没有强烈环境因素干扰(强光、表盘旋转、过度倾斜、室外复杂天气等)
2、精确度要求不高
