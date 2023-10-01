# Leaf-Spot-Analysis-GUI
CAU Course Project Design  for Image Processing Experiments
**一、实验介绍**

**1.1**  **项目背景和意义**

苹果是世界上广泛种植和消费的水果之一，但在生长过程中容易受到各种病害的侵袭，导致产量和品质下降，严重时甚至会危及果树的生命。因此，对苹果病害的及早发现、准确识别和有效治疗是苹果种植业生产的重要保障。

本课程设计旨在利用计算机视觉技术，特别是基于Python的OpenCV库，实现苹果病害叶片的分离和病斑提取分析，以提高病害的检测和诊断效率，降低病害对苹果生产的损失，为苹果种植业提供更好的技术支持。

**1.2**  **项目目标和任务**

**1.2.1**** 项目目标**

·提取图像中的苹果叶片区域。

·计算叶片的总面积。

·从叶片区域中提取叶片的病斑或健康部分。

·计算叶片病斑的面积。

·计算叶片的损伤比例。

**1.2.2**** 项目任务**

图像预处理：

·读取输入图像。

·对图像进行必要的预处理操作，如调整大小、降噪、灰度化等。

·应用适当的阈值方法以获取二值图像。

提取苹果叶片区域：

·使用图像分割方法（如基于阈值、边缘检测等）来提取苹果叶片的区域。

·可以根据颜色、形状、纹理等特征来区分叶片和其他区域。

计算叶片总面积：

·对提取的苹果叶片区域进行处理，计算其像素面积。

·可以使用像素计数或者图像分析方法来实现。

提取叶片病斑或健康部分：

·根据预先定义的病斑特征（颜色、纹理、形状等），从叶片区域中提取病斑或健康部分。

·可以使用阈值分割、形态学操作、区域生长等方法来实现。

计算叶片病斑面积：

·对提取的病斑区域进行处理，计算其像素面积。

计算叶片损伤比例：

·将叶片病斑面积除以叶片总面积，得到叶片的损伤比例。

输出结果：

·将计算得到的叶片总面积、病斑面积和损伤比例等结果进行显示或保存。

**1.3**  **项目流程和方法**

![](RackMultipart20231001-1-okzepl_html_5c04f78495f224a4.png)

**1.4**  **项目实验环境**

操作系统： Windows 10

Python 版本： Python 3.9

运行依赖：

matplotlib

opencv-python

numpy 1.18.5

scikit-image

图形库程序依赖：tkinter

**二、图像预处理**

**2.1**  **原始图像的读取和显示**

本实验选取一个放置在白色背景板上的病害苹果叶片的图像，其余的图像处理部分基于此原始图像。

![](RackMultipart20231001-1-okzepl_html_17490dc9746e79a7.jpg)

图1 原始图像

**2.2**  **图像灰度化和直方图显示**

![](RackMultipart20231001-1-okzepl_html_440d1e1c7a65049a.jpg)

图2 灰度图像

源代码：

![Shape1](RackMultipart20231001-1-okzepl_html_700943fa3815e46d.gif)

img = cv2.imread('AppleDiseaseLeaves/1.jpg')
 gray = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)
 cv2.imwrite('AppleDiseaseLeaves/gray.jpg', gray)

![](RackMultipart20231001-1-okzepl_html_9bc1f0a53c4a18b9.png)

图3 灰度直方图

![Shape2](RackMultipart20231001-1-okzepl_html_251d126448277d74.gif)

# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')
 gray = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)
 plt.hist(gray.ravel(), bins=256, range=[0, 256])
 plt.title('Gray Image Histogram')
 plt.show()

源代码：

**2.3 RGB**** 三通道图和直方图**

将图像的RGB三个通道分离出来，绘制三个通道对应灰度直方图，结果如下图所示：

![](RackMultipart20231001-1-okzepl_html_89c048cfa55c2daa.png)

图4 RGB通道图、RGB通道灰度直方图

![Shape3](RackMultipart20231001-1-okzepl_html_a55cfecf9e3dbeae.gif)

# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')
# 分离通道
b, g, r = cv2.split(img)
# 显示三个通道
plt.figure(figsize=(12, 5))
 plt.subplot(231)
 plt.imshow(b, cmap='gray')
 plt.title('Blue Channel')
 plt.axis('off')
 plt.subplot(232)
 plt.imshow(g, cmap='gray')
 plt.title('Green Channel')
 plt.axis('off')
 plt.subplot(233)
 plt.imshow(r, cmap='gray')
 plt.title('Red Channel')
 plt.axis('off')
# 分离三通道
b,g,r = cv2.split(img)
 # 显示三个通道的直方图
plt.subplot(234)
 plt.hist(b.ravel(), bins=256, color='b', range=[0, 256])
 plt.title('Blue Channel Histogram')
 plt.subplot(235)
 plt.hist(g.ravel(), bins=256, color='g', range=[0, 256])
 plt.title('Green Channel Histogram')
 plt.subplot(236)
 plt.hist(r.ravel(), bins=256, color='r', range=[0, 256])
 plt.title('Red Channel Histogram')
 plt.tight\_layout()
 plt.show()

源代码：

**2.4**  **图像的滤波**

本次实验选用了三种经典的空间域方法：均值滤波、中值滤波和高斯滤波，和频率域的平滑方法：低通高斯滤波。

尝试观察上述几种方法的平滑效果，选择效果最佳的方法。

**空间域方法：**

![](RackMultipart20231001-1-okzepl_html_f47a80732e0743f1.jpg)

图5 高斯滤波图像

![](RackMultipart20231001-1-okzepl_html_45652ea44d53eecf.jpg)

图6 均值滤波图像

![](RackMultipart20231001-1-okzepl_html_f915ba45a26b88cd.jpg)

图7 中值滤波图像

**频率域方法：**

高斯低通滤波：（对灰度图处理）

![](RackMultipart20231001-1-okzepl_html_655343719391b1d2.jpg)

图8 高斯低通滤波图像

在没有特别突出的噪声信号的情况下，均值、高斯、中值滤波和频率域高斯低通滤波都可以用于图像去噪。要选择哪种滤波方法取决于图像的特点以及需要保留的细节。一般来说，高斯滤波在大多数情况下都是一种较好的选择，因为它可以在一定程度上平滑图像并保留细节。考虑图像中并没有明显噪声，主要存在的是细小的高斯噪声，因此这里选用高斯滤波对图像进行去噪。

![Shape4](RackMultipart20231001-1-okzepl_html_3cdefb44655ff60a.gif)

# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')
# 定义卷积核大小
kernel\_size = 5
# 高斯滤波器
img\_gaussian = cv2.GaussianBlur(img, (kernel\_size, kernel\_size), 0)
# 中值滤波器
img\_median = cv2.medianBlur(img, kernel\_size)
# 均值滤波器

# gaussian lowpass filter
def GaussianLowFilter(image, d):
 f = np.fft.fft2(image)
 fshift = np.fft.fftshift(f)

def make\_transform\_matrix(d):
 transfor\_matrix = np.zeros(image.shape)
 center\_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
for i in range(transfor\_matrix.shape[0]):
for j in range(transfor\_matrix.shape[1]):
def cal\_distance(pa, pb):
from math import sqrt
 dis = sqrt((pa[0] - pb[0]) \*\* 2 + (pa[1] - pb[1]) \*\* 2)
return dis

 dis = cal\_distance(center\_point, (i, j))
 transfor\_matrix[i, j] = np.exp(-(dis \*\* 2) / (2 \* (d \*\* 2)))
return transfor\_matrix

 d\_matrix = make\_transform\_matrix(d)
 new\_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift \* d\_matrix)))
return new\_img

img = cv2.imread('AppleDiseaseLeaves/1.jpg')
 r, g, b = cv2.split(img)
 smooth\_b = GaussianLowFilter(b,100)
 smooth\_g = GaussianLowFilter(g,100)
 smooth\_r = GaussianLowFilter(r,100)
 enhanced\_img = cv2.merge((smooth\_r, smooth\_g, smooth\_b))
 cv2.imwrite('DFT/GaussianLowFilter.jpg',enhanced\_img)

源代码：

**2.4**  **图像的锐化**

本次实验选用了空间域上锐化方案：Robert算子、Sobel算子、Prewitt算子、Laplacian算子和LoG算子5种算子，还有频率域锐化方案：高斯高通滤波。分别对灰度图进行以下6种方案的边缘提取， 6种作用后的边缘图像如下所示：

![](RackMultipart20231001-1-okzepl_html_caf8bcb996ca5fa2.png) ![](RackMultipart20231001-1-okzepl_html_1653777d773fbb99.png)

图9 Robert算子边缘检测图10 Sobel算子边缘检测

![](RackMultipart20231001-1-okzepl_html_af516642091ed018.png) ![](RackMultipart20231001-1-okzepl_html_b1e597508a7d7eb8.png)

图11 Prewitt算子边缘检测图12 Laplacian算子边缘检测

![](RackMultipart20231001-1-okzepl_html_b8dacc7be1546cb0.png) ![](RackMultipart20231001-1-okzepl_html_2f50379b0f8d37ac.jpg)

图13 LoG算子边缘检测图14 高斯高通滤波边缘检测

经过比较发现，Robert、Prewitt检测的边缘存在大量噪点，Laplacian、LoG和高斯高通滤波边缘较为模糊，经过慎重考虑后，选择使用Sobel算子进行处理。

Sobel对原图的RGB通道进行边缘提取后，与原图权重叠加获得锐化图像，尝试观察6种方法的锐化效果。

![](RackMultipart20231001-1-okzepl_html_e2228c0d71cf8e71.png)

图15 6种锐化算法效果图

经过慎重考虑，最终选择使用Sobel算子进行图像的锐化操作。

![Shape5](RackMultipart20231001-1-okzepl_html_816a4cb1b3352bc4.gif)

def roberts\_edge\_detection(img):
# 定义Roberts算子
roberts\_x = np.array([[-1, 0], [0, 1]])
 roberts\_y = np.array([[0, -1], [1, 0]])
# 对图像进行卷积操作
grad\_x = cv2.filter2D(img, -1, roberts\_x)
 grad\_y = cv2.filter2D(img, -1, roberts\_y)
# 计算梯度幅值
grad = np.sqrt(grad\_x \*\* 2 + grad\_y \*\* 2)
# 返回边缘检测结果
return grad.astype(np.uint8)
def sobel\_edge\_detection(img, ksize=3):
# 使用Sobel算子进行边缘检测
grad\_x = cv2.Sobel(img, cv2.CV\_64F, 1, 0, ksize=ksize)
 grad\_y = cv2.Sobel(img, cv2.CV\_64F, 0, 1, ksize=ksize)
# 计算梯度幅值
grad = np.sqrt(grad\_x \*\* 2 + grad\_y \*\* 2)
# 返回边缘检测结果
return grad.astype(np.uint8)
def prewitt\_edge\_detection(img):
# 使用Prewitt算子进行边缘检测
kernel\_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
 kernel\_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
 grad\_x = cv2.filter2D(img, -1, kernel\_x)
 grad\_y = cv2.filter2D(img, -1, kernel\_y)
# 计算梯度幅值
grad = np.sqrt(grad\_x \*\* 2 + grad\_y \*\* 2)
# 返回梯度幅值
return grad.astype(np.uint8)
def laplacian\_edge\_detection(img, ksize=3):
# 使用Laplacian算子进行边缘检测
laplacian = cv2.Laplacian(img, cv2.CV\_64F, ksize=ksize)
# 计算梯度幅值
grad = np.abs(laplacian)
return grad.astype(np.uint8)
def laplacian\_of\_gaussian\_edge\_detection(img, ksize=3, sigma=1.0):
# 使用高斯模糊对图像进行平滑处理
img\_smooth = cv2.GaussianBlur(img, (ksize, ksize), sigma)
# 使用Laplacian算子进行边缘检测
laplacian = cv2.Laplacian(img\_smooth, cv2.CV\_64F, ksize=ksize)
# 计算梯度幅值
grad = np.abs(laplacian)
return grad.astype(np.uint8)

源代码：

![Shape6](RackMultipart20231001-1-okzepl_html_f3d0548a16fd6df5.gif)

# gaussian highpass
def GaussianHighFilter(image, d):
 f = np.fft.fft2(image)
 fshift = np.fft.fftshift(f)
def make\_transform\_matrix(d):
 transfor\_matrix = np.zeros(image.shape)
 center\_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
for i in range(transfor\_matrix.shape[0]):
for j in range(transfor\_matrix.shape[1]):
def cal\_distance(pa, pb):
from math import sqrt
 dis = sqrt((pa[0] - pb[0]) \*\* 2 + (pa[1] - pb[1]) \*\* 2)
return dis
 dis = cal\_distance(center\_point, (i, j))
 transfor\_matrix[i, j] = 1 - np.exp(-(dis \*\* 2) / (2 \* (d \*\* 2)))
return transfor\_matrix
 d\_matrix = make\_transform\_matrix(d)
 new\_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift \* d\_matrix)))
return new\_img

**三、苹果叶片的分离**

**3.1**  **阈值分割方案**

**3.1.1 RGB**** 、 ****HSV**** 多通道图比较**

![](RackMultipart20231001-1-okzepl_html_9c5157ad348b02f2.jpg) ![](RackMultipart20231001-1-okzepl_html_f866ae84ef0b6cf3.jpg) ![](RackMultipart20231001-1-okzepl_html_fe9125b309fa912d.jpg)

图16 R通道图17 G通道图18 B通道

![](RackMultipart20231001-1-okzepl_html_c8fbd15fd6435086.jpg) ![](RackMultipart20231001-1-okzepl_html_76591d08b0a44e1e.jpg) ![](RackMultipart20231001-1-okzepl_html_9c44dcc7f3517863.jpg)

图19 H通道图20 S通道图21 V通道

可以看到，S通道、R通道在叶片上叶片和背景的区分度明显，优先选择S通道和R通道进行阈值分割。

源代码：

![Shape7](RackMultipart20231001-1-okzepl_html_8d375101f54a4502.gif)

import cv2
import matplotlib.pyplot as plt
import os
# 创建文件夹
if not os.path.exists('HSV'):
 os.makedirs('HSV')
# 读取图像
img = cv2.imread('sharpening/enhanced\_img\_sobel.jpg')
# 将图像从RGB空间转换为HSV空间
hsv = cv2.cvtColor(img, cv2.COLOR\_BGR2HSV)
# 提取H、S、V三个通道的图像
h, s, v = cv2.split(hsv)
# 导出H通道图像
cv2.imwrite('HSV/h\_channel.jpg', h)
# 导出S通道图像
cv2.imwrite('HSV/s\_channel.jpg', s)
# 导出V通道图像
cv2.imwrite('HSV/v\_channel.jpg', v)
# 设置直方图参数
hist\_bins = 256
hist\_range = (0, 256)
# 计算H、S、V三个通道的直方图
hist\_h = cv2.calcHist([h], [0], None, [hist\_bins], hist\_range)
 hist\_s = cv2.calcHist([s], [0], None, [hist\_bins], hist\_range)
 hist\_v = cv2.calcHist([v], [0], None, [hist\_bins], hist\_range)
# 在plt窗口显示H、S、V三个通道的直方图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
 ax1.hist(h.ravel(), bins=hist\_bins, range=hist\_range)
 ax1.set\_title('Hue Histogram')
 ax2.hist(s.ravel(), bins=hist\_bins, range=hist\_range)
 ax2.set\_title('Saturation Histogram')
 ax3.hist(v.ravel(), bins=hist\_bins, range=hist\_range)
 ax3.set\_title('Value Histogram')
 plt.show()

RGB三通道的源代码已经在前面给出，此处省略。

**3.1.2**  **阈值分割方法**

对每个通道使用大津OTSU阈值法自动计算阈值，得到的二值图像如下：

![](RackMultipart20231001-1-okzepl_html_4c2f03a0b00a61e2.png)

图22 6通道使用大津法阈值分割

可见，R和S通道的叶片分离最为完整。其中，R通道的叶片边缘有些许断裂，而S通道不存在这个问题，但是S通道存在比较大的斑点。

现在，对S通道取反后，与R通道进行或运算，得到的图片如下：

![](RackMultipart20231001-1-okzepl_html_5057896897429daa.jpg)

图23 R和S通道运算结果

可以看到已经基本消除叶片边缘上的断裂，只剩下一些细小的噪点，接下来使用形态学操作进行区域填充解决这个问题。

![Shape8](RackMultipart20231001-1-okzepl_html_aeddd967538b1b4f.gif)

def THRESH\_OTSU():
hsv = cv2.cvtColor(img, cv2.COLOR\_BGR2HSV) # 转换为HSV空间
h, s, v = cv2.split(hsv) # 分离通道
 r, g, b = cv2.split(img)
otsu\_thresh = cv2.THRESH\_OTSU # OTSU阈值法
\_, h\_th = cv2.threshold(h, 0, 255, otsu\_thresh)
 \_, s\_th = cv2.threshold(s, 0, 255, otsu\_thresh)
 \_, v\_th = cv2.threshold(v, 0, 255, otsu\_thresh)
 \_, r\_th = cv2.threshold(r, 0, 255, otsu\_thresh)
 \_, g\_th = cv2.threshold(g, 0, 255, otsu\_thresh)
 \_, b\_th = cv2.threshold(b, 0, 255, otsu\_thresh)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
 axes[0, 0].imshow(h\_th, cmap='gray')
 axes[0, 0].set\_title('Hue Thresholding')
 axes[0, 0].axis('off')
 axes[0, 1].imshow(s\_th, cmap='gray')
 axes[0, 1].set\_title('Saturation Thresholding')
 axes[0, 1].axis('off')
 axes[0, 2].imshow(v\_th, cmap='gray')
 axes[0, 2].set\_title('Value Thresholding')
 axes[0, 2].axis('off')
 axes[1, 0].imshow(r\_th, cmap='gray')
 axes[1, 0].set\_title('Red Thresholding')
 axes[1, 0].axis('off')
 axes[1, 1].imshow(g\_th, cmap='gray')
 axes[1, 1].set\_title('Green Thresholding')
 axes[1, 1].axis('off')
 axes[1, 2].imshow(b\_th, cmap='gray')
 axes[1, 2].set\_title('Blue Thresholding')
 axes[1, 2].axis('off')
 plt.tight\_layout()
 plt.show()
 THRESH\_OTSU()
hsv = cv2.cvtColor(img, cv2.COLOR\_BGR2HSV) # 转换为HSV空间
h, s, v = cv2.split(hsv) # 分离通道
 r, g, b = cv2.split(img)
 otsu\_thresh = cv2.THRESH\_OTSU
 \_, s\_th = cv2.threshold(s, 0, 255, otsu\_thresh)
 \_, r\_th = cv2.threshold(r, 0, 255, otsu\_thresh)
s\_th = cv2.bitwise\_not(s\_th) # 取反
result = cv2.bitwise\_or(r\_th, s\_th) # 叠加
 cv2.imwrite('ThresholdSegmentation/Origin.jpg', result)

源代码：

**3.1.3**  **形态学操作和区域填充**

经过反复测试，选择合适的结构元素后，对阈值分割后的图像进行一次开运算后，噪点完全消失，但是边缘细节有些许损失，再进行一次闭运算改善部分细节后，使用掩模运算可以提取出比较完整的叶片。

![](RackMultipart20231001-1-okzepl_html_8c8cda14eed98044.png)

图24 形态学操作过程

![](RackMultipart20231001-1-okzepl_html_97214c6f8f2c188f.jpg)

图25 叶片阈值分割结果

![Shape9](RackMultipart20231001-1-okzepl_html_dbd4dffee77d501b.gif)

# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (9, 9))
# 显示二值化结果
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6))
 axes[0].imshow(result, cmap='gray')
 axes[0].axis('off')
 axes[0].set\_title('Origin')
# 进行开运算操作
result\_open = cv2.morphologyEx(result, cv2.MORPH\_OPEN, kernel)
# 显示结果
axes[1].imshow(result\_open, cmap='gray')
 axes[1].axis('off')
 axes[1].set\_title('First Open operation')
# 进行闭运算操作
kernel = cv2.getStructuringElement(cv2.MORPH\_ELLIPSE, (9, 9))
 closed = cv2.morphologyEx(result\_open, cv2.MORPH\_CLOSE, kernel)
 cv2.imwrite('ThresholdSegmentation/SegmentationOperation.jpg', closed)
# 显示结果
axes[2].imshow(closed, cmap='gray')
 axes[2].axis('off')
 axes[2].set\_title('Second Close operation')
'''
图像Mask进行分割
 '''
Mask = cv2.bitwise\_not(closed)
# 将mask三通道扩展为与img相同的通道数
MergeMask = cv2.merge([Mask, Mask, Mask])
# 将img与mask相乘，保留mask为255的像素，其他像素置为0
masked\_img = cv2.bitwise\_and(img, MergeMask)
 cv2.imwrite('ThresholdSegmentation/LeafSegmentation.jpg', masked\_img)
# 显示结果
axes[3].imshow(cv2.cvtColor(masked\_img, cv2.COLOR\_BGR2RGB))
 axes[3].axis('off')
 axes[3].set\_title('Segmentation')
 plt.tight\_layout()
 plt.show()

源代码：

**3.2**  **边缘检测填充方案**

**3.2.1 Canny**** 算子检测边缘**
直接使用canny算子获取边缘，出现边缘断裂/内部噪点边的问题，调整Canny算子阈值效果也不佳。

![](RackMultipart20231001-1-okzepl_html_e752287a465a2d38.jpg)

图26 Canny边缘检测结果

![Shape10](RackMultipart20231001-1-okzepl_html_8ee23408e13be904.gif)

# 读取图像
img = cv2.imread('sharpening/enhanced\_img\_sobel.jpg')
 fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
# 转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)
# 应用高斯滤波器平滑图像
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 使用Canny算子进行边缘检测
edges = cv2.Canny(blur, 80, 150)
 cv2.imwrite('cannySeg/CannyDetect.jpg', edges)
 ax[0, 0].imshow(edges, cmap='gray')
 ax[0, 0].axis('off')
 ax[0, 0].set\_title('Canny edge Detect')

源代码：

**3.2.2**  **形态学操作连接边缘**

使用形态学膨胀连接断裂的边缘，利于查找轮廓算法的实现。

![](RackMultipart20231001-1-okzepl_html_1c32eb9af96bd005.jpg)

图27 形态学膨胀结果

![Shape11](RackMultipart20231001-1-okzepl_html_fe2b85a065556cb.gif)

# 进行形态学膨胀处理
kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (5, 10))
 dilated = cv2.dilate(edges, kernel, iterations=2)
# 显示结果
cv2.imwrite('cannySeg/dilated.jpg', dilated)
 ax[0, 1].imshow(dilated, cmap='gray')
 ax[0, 1].axis('off')
 ax[0, 1].set\_title('Morphological dilate processing')

源代码：

**3.2.2**  **查找轮廓和区域填充**

查找轮廓后填充内部区域，使用腐蚀减小膨胀对边缘的影响。

![](RackMultipart20231001-1-okzepl_html_ac23bf9ef5e2e279.jpg)

图28 查找轮廓和区域填充结果

![](RackMultipart20231001-1-okzepl_html_4c7d8313458657d9.jpg)

图29 叶片分割结果

![Shape12](RackMultipart20231001-1-okzepl_html_f479c435e7553c78.gif)

# 找到边缘所围成的轮廓
contours, \_ = cv2.findContours(dilated, cv2.RETR\_EXTERNAL, cv2.CHAIN\_APPROX\_SIMPLE)
# 创建空图像，大小与输入图像相同
mask = np.zeros\_like(img)
# 填充轮廓
cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED)
# 定义腐蚀结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (5, 10))
# 进行腐蚀处理
mask = cv2.erode(mask, kernel, iterations=2)
 cv2.imwrite('cannySeg/mask.jpg', mask)
# 显示结果
ax[1, 0].imshow(mask)
 ax[1, 0].axis('off')
 ax[1, 0].set\_title('filled and erode')

cv2.imwrite('cannySeg/result.jpg', cv2.bitwise\_and(mask, img))
 result = cv2.cvtColor(cv2.bitwise\_and(mask, img), cv2.COLOR\_BGR2RGB)
# 显示结果
ax[1,1].imshow(result)
 ax[1,1].axis('off')
 ax[1, 1].set\_title('canny seg result')
 plt.tight\_layout()
 plt.show()

源代码：

**四、提取图像的特征**

**4.1**  **图像的形状特征**

读取阈值分割后的叶片掩模图像，进行形状特征的分析。

![](RackMultipart20231001-1-okzepl_html_1b59f6592f8198a4.jpg)

图30 阈值分割掩模图像

查找边缘和轮廓，计算叶片图像的形状特征。

![](RackMultipart20231001-1-okzepl_html_38909537b06483.png)

图31 控制台打印图像形状特征

![](RackMultipart20231001-1-okzepl_html_b28b8c2b5d4429be.png)

图32 外接矩形和最小外接矩形

![Shape13](RackMultipart20231001-1-okzepl_html_eaa2ce63b61a7972.gif)

img = cv2.imread('ThresholdSegmentation/SegmentationOperation.jpg')
 img = cv2.bitwise\_not(img)
# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR\_BGR2GRAY)
# 二值化处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH\_BINARY)
# 获取轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR\_TREE, cv2.CHAIN\_APPROX\_SIMPLE)
# 计算面积和周长
area = cv2.contourArea(contours[0])
 perimeter = cv2.arcLength(contours[0], True)
# 计算矩形
x,y,w,h = cv2.boundingRect(contours[0])
# 计算圆形度
circularity = 4\*np.pi\*area/(perimeter\*\*2)
# 计算重心
M = cv2.moments(contours[0])
 cx = int(M['m10']/M['m00'])
 cy = int(M['m01']/M['m00'])

源代码：

![Shape14](RackMultipart20231001-1-okzepl_html_a7b2b0adca2c9e85.gif)

# 计算复杂度
e = (perimeter \*\* 2) / area
# 输出结果
print('面积:', area)
print('周长:', perimeter)
print('长度:', w)
print('宽度:', h)
print('圆形度:', circularity)
print('重心:', cx, cy)
print('复杂度:', e)
 img = cv2.imread('AppleDiseaseLeaves/1.jpg')
 first\_img = img.copy()
# 绘制矩形和重心
cv2.rectangle(first\_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
 cv2.circle(first\_img, (cx, cy), 5, (0, 0, 255), -1)
# 计算最小外接矩形
rect = cv2.minAreaRect(contours[0])
 box = cv2.boxPoints(rect)
 box = np.int0(box)
 Second\_img = img.copy()
# 计算最小外接矩形的面积
rect\_area = cv2.contourArea(box)
print('最小外接矩形面积为:', rect\_area)
print('矩形度:', area/rect\_area)
# 在图像上绘制最小外接矩形
cv2.drawContours(Second\_img, [box], 0, (0, 0, 255), 2)
# 计算最小外接矩形的重心
M = cv2.moments(box)
 cx = int(M['m10']/M['m00'])
 cy = int(M['m01']/M['m00'])
print("最小外接矩形重心坐标：({}, {})".format(cx, cy))
# 在图像上绘制重心
cv2.drawMarker(Second\_img, (cx, cy), (255, 0, 0), markerType=cv2.MARKER\_CROSS, thickness=2)
# 生成空白图片
third\_img = np.zeros\_like(img)
# 绘制轮廓
cv2.drawContours(third\_img, contours, -1, (0, 255, 0), 2)

**4.2**  **灰度共生矩阵和纹理特征**

对完整的叶片图像计算灰度共生矩阵，选择灰度共生矩阵统计量有5种：对比度、差异性、均匀度、能量、相关性。

分析10张病害苹果叶片分析所得结果：

![](RackMultipart20231001-1-okzepl_html_d7de59f78f2c931d.png)

图33 控制台打印统计量结果

对比度（Contrast）：对比度值在范围4.0967到11.0627之间。对比度指标表示图像中相邻像素之间的灰度差异程度。较高的对比度值意味着图像中的相邻像素之间具有较大的灰度差异。

差异性（Dissimilarity）：差异性值在范围1.1524到1.9529之间。差异性指标表示图像中相邻像素之间的灰度差异程度的加权平均。较高的差异性值表示图像中的相邻像素之间的灰度差异较大。

均匀度（Homogeneity）：均匀度值在范围0.4513到0.5567之间。均匀度指标表示图像中相邻像素灰度值之间的相似程度。较高的均匀度值表示图像中的相邻像素之间的灰度值较为相似。

能量（Energy）：能量值在范围0.0376到0.0641之间。能量指标表示图像中相邻像素灰度值的概率分布的均匀程度。较高的能量值表示图像中的相邻像素的灰度分布更加均匀。

相关性（Correlation）：相关性值在范围0.9984到0.9995之间。相关性指标表示图像中相邻像素之间的线性关系程度。较高的相关性值表示图像中的相邻像素之间存在较强的线性关系。

由此得到一些如下的分析结果：

对比度和差异性值的变化范围较大，可能反映了病害区域与健康区域、背景的灰度差异明显。

均匀度值较高，说明叶片大部分区域内的像素灰度值相对均匀，没有明显的灰度跳变。

能量值较低，可能表示病害区域内的像素灰度分布相对不均匀，存在一些灰度值的集中。

相关性值接近1，可能说明病害区域内的像素之间存在较强的线性关系。

![](RackMultipart20231001-1-okzepl_html_6f81c09e0157c009.png)

图34 统计量数据可视化

![Shape15](RackMultipart20231001-1-okzepl_html_93b4904f24727e2b.gif)

# 定义函数计算纹理特征
def compute\_texture\_features(image\_path):
# 读取图像
img = io.imread(image\_path)
# 转换成灰度图像
gray = color.rgb2gray(img)
# 计算灰度共生矩阵
glcm = graycomatrix(np.uint8(gray \* 255), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
# 计算能量、对比度、相关性
contrast = graycoprops(glcm, 'contrast') # 对比度
dissimilarity = graycoprops(glcm, 'dissimilarity') # 差异性
homogeneity = graycoprops(glcm, 'homogeneity') # 均匀度
energy = graycoprops(glcm, 'energy') # 能量/角二阶矩
correlation = graycoprops(glcm, 'correlation') # 相关性
return contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0]
# 定义函数绘制五边形图
def plot\_pentagon(features, feature\_names):
 fig = plt.figure(figsize=(8, 7))
 angles = np.linspace(0, 2\*np.pi, len(feature\_names) + 1, endpoint=True)
 ax = fig.add\_subplot(111, polar=True)
 plt.xticks(angles[:-1], feature\_names)
 ax.plot(angles, [1]\*len(angles), 'k--')
for i, f in enumerate(features):
 values = np.array(f[:5])
 min\_val = min(values)
 max\_val = max(values)
 values = (values - min\_val) / (max\_val - min\_val)
 ax.set\_ylim()
 ax.plot(angles[:-1], values, linewidth=2, label="Image "+str(i+1))
 ax.fill(angles[:-1], values, alpha=0.25)
 ax.legend(loc='upper left', bbox\_to\_anchor=(0.1, 0.1))
 plt.tight\_layout()
 plt.show()

源代码：

**五、苹果病斑的提取和分析**

**5.1**  **阈值分割方案**

**5.1.1**  **图像分割和目标提取**

视察RGB通道和HSV通道图像，发现R通道图像对病斑和叶片具有比较明显的区分度，对其计算灰度直方图，属于单峰图像，很容易观察出，病斑属于亮度较高的像素，因此选择合适的阈值，可以很容易的把病斑从叶片中分离出来。

![](RackMultipart20231001-1-okzepl_html_d6cf608aa41b8e38.png)

图35 R通道掩模图像和直方图

手动设定阈值， 发现threshold\_value = 99下病斑保留比较完整又尽可能减少噪点。

![](RackMultipart20231001-1-okzepl_html_a605b48a179f1f88.jpg) ![](RackMultipart20231001-1-okzepl_html_3ccfa104811e0944.jpg)

图36 原始图像图37 阈值分割

![Shape16](RackMultipart20231001-1-okzepl_html_6b12694dad006a85.gif)

# 读取图像
img = cv2.imread('RGB/r\_channel.jpg')
# 读入叶片分割二值图像
mask = cv2.imread('ThresholdSegmentation/SegmentationOperation.jpg')
# 取反
mask = cv2.bitwise\_not(mask)
# 对图像进行掩膜处理
masked\_img = cv2.bitwise\_and(img, mask)

# 设定阈值
threshold\_value = 99
# 阈值分割
ret, thresh = cv2.threshold(masked\_img, threshold\_value, 255, cv2.THRESH\_BINARY)
# 显示分割后二值化图像
ax[1,0].imshow(cv2.cvtColor(thresh, cv2.COLOR\_BGR2RGB))
 ax[1,0].axis('off')
 ax[1,0].set\_title('thresh process')
 cv2.imwrite('LesionThreshSeg/threshprocess.jpg', thresh)

源代码：

**5.1.2**  **形态学操作**

阈值分割结果存在大量长条状的叶脉和噪声，因此可以考虑使用长方形的条状结构元素进行开运算以消除叶脉，然后使用小的结构元素进行开运算继续消除细小的噪声。

![](RackMultipart20231001-1-okzepl_html_b824e266e6aff630.jpg)

图38 经过开运算和闭运算后的病斑区域

经过开运算出来后我们发现噪声已经基本消失，但是仍然存在一些问题，即病斑内部的褐色斑点被忽视了，这时我们采用闭运算填充内部区域，或者使用查找边缘进行填充解决这个问题。

![](RackMultipart20231001-1-okzepl_html_37a0e64f03b3084b.jpg)

图39 经过开运算和闭运算后的病斑区域

进行掩模运算后即可获取病斑分割图像：

![](RackMultipart20231001-1-okzepl_html_636603e513b20201.jpg)

图40 掩模的病斑区域

计算病斑区域面积：

![](RackMultipart20231001-1-okzepl_html_6d652c7a93c7e9bd.jpg)

图41 病斑轮廓

![](RackMultipart20231001-1-okzepl_html_ca8a1b45599bcfe0.png)

图42 病斑占据区域面积和比例

![Shape17](RackMultipart20231001-1-okzepl_html_5e935a7d9b00e9e1.gif)

'''
开运算消除大部分叶脉
 '''
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (1,15))
# 对图像进行开运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH\_OPEN, kernel)
'''
开运算消除剩余噪点
 '''
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_ELLIPSE, (5, 5))
# 对图像进行开运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH\_OPEN, kernel)
 cv2.imwrite('LesionThreshSeg/open operation.jpg', thresh)
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_ELLIPSE, (40, 40))
# 对图像进行闭运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH\_CLOSE, kernel)
# 显示分割后二值化图像
ax[1,1].imshow(cv2.cvtColor(thresh, cv2.COLOR\_BGR2RGB))
 ax[1,1].axis('off')
 ax[1,1].set\_title('open and close operation')
 cv2.imwrite('LesionThreshSeg/open and close operation.jpg', thresh)

tmp = np.uint8(cv2.cvtColor(thresh,cv2.COLOR\_BGR2GRAY))
# 获取轮廓
contours, hierarchy = cv2.findContours(tmp, cv2.RETR\_TREE, cv2.CHAIN\_APPROX\_SIMPLE)
# 计算所有连通分量的面积总和
total\_area = 0
for contour in contours:
 area = cv2.contourArea(contour)
 total\_area += area
print("连通分量的面积总和：", total\_area)
print("病斑占比：", total\_area/1051978.0)

源代码：

![Shape18](RackMultipart20231001-1-okzepl_html_d4884adca74c4862.gif)

# 生成空白图片
third\_img = np.zeros\_like(img)
# 绘制轮廓
cv2.drawContours(third\_img, contours, -1, (0, 255, 0), 2)
 cv2.imwrite('LesionThreshSeg/contour.jpg', third\_img)
# 对图像进行掩膜处理
Disease\_img = cv2.bitwise\_and(leaf, thresh)
# 显示分割后叶片病斑图像
ax[1,2].imshow(cv2.cvtColor(Disease\_img, cv2.COLOR\_BGR2RGB))
 ax[1,2].axis('off')
 ax[1,2].set\_title('Disease Seg')
 cv2.imwrite('LesionThreshSeg/Disease Seg.jpg', Disease\_img)
 plt.tight\_layout()
 plt.show()

**5.2 Kmeans**** 色彩空间聚类方案**

**5.2.1**  **聚类生成伪彩色图像**

通过观察叶片分离后的原图，我们可以发现，叶片，病斑，背景三者之间具有较大的颜色差异。

![](RackMultipart20231001-1-okzepl_html_97214c6f8f2c188f.jpg)

图43 叶片分离原图

因此我们可以思考使用Kmeans聚类，对空间中所有像素点进行聚簇，设置到合适的簇数时，病斑自然而然会被划分为一个簇。为每一簇的像素集合赋予一个颜色生成伪彩色图像，我们可以清楚的看到，这张图中病斑是黑色的像素区域。

![](RackMultipart20231001-1-okzepl_html_79be942cb4d07b95.jpg)

图44 Kmeans聚类结果

![Shape19](RackMultipart20231001-1-okzepl_html_24581273766723b9.gif)

img = cv2.imread('ThresholdSegmentation/LeafSegmentation.jpg')
# 转换为RGB颜色空间
img = cv2.cvtColor(img, cv2.COLOR\_BGR2RGB)
 colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 255, 255], [0, 0, 0]], dtype=np.float32)
 height, width = img.shape[:2]
 k = 7

# 将像素点进行reshape，以方便使用聚类算法
img\_reshaped = img.reshape(height \* width, 3)
# 使用KMeans算法进行聚类，将图像分为k个类别
kmeans = KMeans(n\_clusters=k, random\_state=0).fit(img\_reshaped)
# 根据聚类结果对图像进行重构
new\_img = np.zeros\_like(img\_reshaped)
for i in range(k):
 new\_img[kmeans.labels\_ == i] = colors[i]
 new\_img = new\_img.reshape(height, width, 3)

源代码：

**5.2.2**  **提取病斑对应的簇像素集合**

上图的伪彩色图像中，病斑被分为黑色的像素点，从黑色像素点中提取病斑图像：

![](RackMultipart20231001-1-okzepl_html_7bc66176507c05a8.jpg)

图45 病斑提取结果

![Shape20](RackMultipart20231001-1-okzepl_html_9365c2661ebae247.gif)

# 提取黑色像素
black\_pixels = np.all(new\_img == [0, 0, 0], axis=-1)
 copy\_img = np.zeros\_like(new\_img)
# 将黑色像素设置为白色，其他像素设置为黑色
copy\_img[black\_pixels] = [255, 255, 255]
 copy\_img[~black\_pixels] = [0, 0, 0]

 cv2.imwrite('RGBKmeansCluster/Seg.jpg', copy\_img)
# 显示结果
plt.imshow(cv2.cvtColor(copy\_img, cv2.COLOR\_BGR2RGB))
 plt.axis('off')
 plt.show()

源代码：

**5.2.3**  **形态学操作**

选择合适的结构元素进行开运算消除噪点。

![](RackMultipart20231001-1-okzepl_html_bbf81ef538ee8221.jpg)

图46 开运算结果

选择合适的结构元素进行闭运算消除病斑断裂的边缘，方便下一步查找边缘并填充。

![](RackMultipart20231001-1-okzepl_html_358333057f94f586.jpg)

图47 闭运算结果

查找边缘，填充内部黑色区域。

![](RackMultipart20231001-1-okzepl_html_7b4685305ef26418.jpg)

图48 查找边缘结果

![](RackMultipart20231001-1-okzepl_html_84e79482a1bcc2c6.jpg)

图49 区域填充结果

![](RackMultipart20231001-1-okzepl_html_e9d28b7754a5ab8a.png)

图50 病斑占据区域面积和比例

![Shape21](RackMultipart20231001-1-okzepl_html_7bc429222bb884e7.gif)

# 读取图像
img = cv2.imread('RGBKmeansCluster/Seg.jpg')
 fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_ELLIPSE, (8,7))
# 对图像进行开运算
thresh = cv2.morphologyEx(img, cv2.MORPH\_OPEN, kernel)
 cv2.imwrite('RGBKmeansCluster/SegOpenOP.jpg', thresh)
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH\_RECT, (5,15))
# 对图像进行闭运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH\_CLOSE, kernel)
 cv2.imwrite('RGBKmeansCluster/SegCloseOP.jpg', thresh)
# 转换为灰度图像
gray = cv2.cvtColor(thresh, cv2.COLOR\_BGR2GRAY)
# 查找边缘
contours, hierarchy = cv2.findContours(gray, cv2.RETR\_EXTERNAL, cv2.CHAIN\_APPROX\_SIMPLE)
# 计算所有连通分量的面积总和
total\_area = 0
for contour in contours:
 area = cv2.contourArea(contour)
 total\_area += area
print("连通分量的面积总和：", total\_area)
print("病斑占比：", total\_area/1051978.0)
# 生成空白图片
third\_img = np.zeros\_like(img)
# 绘制轮廓
cv2.drawContours(third\_img, contours, -1, (0, 255, 0), 2)
 cv2.imwrite('RGBKmeansCluster/contour.jpg', third\_img)
# 绘制填充后的图像
filled\_img = np.zeros\_like(thresh)
 cv2.drawContours(filled\_img, contours, -1, (255, 255, 255), thickness=-1)
 cv2.imwrite('RGBKmeansCluster/SegFilled.jpg', filled\_img)
 ax.imshow(filled\_img)

源代码：

**5.3**  **区域生长法剔除病斑**

采用区域生长法，选中若干初始生长点进行运算，获取剔除病斑的部分，效果不如前面两种方案。

![](RackMultipart20231001-1-okzepl_html_4bc9003087155b8f.jpg)

图51 健康叶片掩模

![](RackMultipart20231001-1-okzepl_html_e0a1ef4c979d0308.jpg)

图52 健康叶片区域

可以看到，叶片中心的部分仍然存在一部分噪声，可以考虑进一步进行形态学处理。使用形态学闭运算消除一部分噪声，但无法做到完全消除。

![](RackMultipart20231001-1-okzepl_html_fed6f8e03393485b.jpg)

图53 形态学操作结果

![Shape22](RackMultipart20231001-1-okzepl_html_2a64935fa58e5a9.gif)

# 提取H、S、V三个通道的图像
b,g,r = cv2.split(img)
 gray\_img = r
 h, w = gray\_img.shape[:2]
def get\_binary\_img(img):
# gray img to bin image
bin\_img = np.zeros(shape=(img.shape), dtype=np.uint8)
 h = img.shape[0]
 w = img.shape[1]
for i in range(h):
for j in range(w):
 bin\_img[i][j] = 255 if img[i][j] \> 127 else 0
return bin\_img
bin\_img = get\_binary\_img(gray\_img)
 out\_img = np.zeros(shape=(bin\_img.shape), dtype=np.uint8)
# 选择初始3个种子点
seeds = [(887, 833), (1049, 989)]
for seed in seeds:
 x = seed[0]
 y = seed[1]
 out\_img[y][x] = 255

源代码：

![Shape23](RackMultipart20231001-1-okzepl_html_8f22b3b6cc45c9ea.gif)

# 8 邻域
directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
 visited = np.zeros(shape=(bin\_img.shape), dtype=np.uint8)
while len(seeds):
 seed = seeds.pop(0)
 x = seed[0]
 y = seed[1]
# visit point (x,y)
visited[y][x] = 1
for direct in directs:
 cur\_x = x + direct[0]
 cur\_y = y + direct[1]
if cur\_x \< 0 or cur\_y \< 0 or cur\_x \>= w or cur\_y \>= h:
continue
# 没有访问过且属于同一目标
if (not visited[cur\_y][cur\_x]) and (bin\_img[cur\_y][cur\_x] == bin\_img[y][x]):
 out\_img[cur\_y][cur\_x] = 255
visited[cur\_y][cur\_x] = 1
seeds.append((cur\_x, cur\_y))
# 定义行数和列数
rows = 1
cols = 2
# 创建一个大小为(15,8) 的窗口，用来显示多个子图
fig, ax = plt.subplots(rows, cols, figsize=(15, 8))
 cv2.imwrite('RegionGrowing/RegionGrowingMask.jpg', out\_img)
 ax[0].imshow(out\_img,cmap='gray')
 ax[0].set\_title('mask')
 ax[0].axis('off')
 bake\_img = cv2.imread('AppleDiseaseLeaves/1.jpg')
 h = bake\_img.shape[0]
 w = bake\_img.shape[1]
for i in range(h):
for j in range(w):
if out\_img[i][j] == 255:
 bake\_img[i][j][0] = 0
bake\_img[i][j][1] = 0
bake\_img[i][j][2] = 0
# 可视化分割结果
cv2.imwrite('RegionGrowing/RegionGrowingResult.jpg', bake\_img)
 bake\_img = cv2.cvtColor(bake\_img, cv2.COLOR\_BGR2RGB)
 ax[1].imshow(bake\_img)
 ax[1].set\_title('object')
 ax[1].axis('off')
 plt.show()

**六、**** GUI ****图形界面**

本次实验撰写了部分算法在GUI上的实现，可以更加直观地进行部分操作而不用频繁修改算法。

![](RackMultipart20231001-1-okzepl_html_286e65aa44b7f72f.png)

开始界面

![](RackMultipart20231001-1-okzepl_html_b72a23d646a40652.png)

打开图片界面

灰度图为设置当前彩色图像为灰度图，以及查看当前图的灰度直方图。

![](RackMultipart20231001-1-okzepl_html_b890d135e3320594.png)

彩色通道界面

色彩空间可以查看任意一个通道的灰度图和对应的灰度直方图。

![](RackMultipart20231001-1-okzepl_html_3a8ead8f22185dcb.png)

平滑操作界面

平滑可以对灰度图和RGB图像进行处理。其中频率域处理较慢会有提示。

![](RackMultipart20231001-1-okzepl_html_f1b497de55a4b1ef.png)

锐化操作界面

锐化可以对灰度图和RGB图像进行处理。其中频率域处理较慢会有提示。

![](RackMultipart20231001-1-okzepl_html_d93ff6398851c890.png)

边缘检测界面

边缘检测可以使用一些预设的算法进行处理。频率域处理较慢会有提示。

![](RackMultipart20231001-1-okzepl_html_bdce618e7746db04.png)

阈值分割-叶片界面

阈值分割会有一些处理方法。大津法可以查看当前RGB图像所有通道的大津法处理结果。

预设则给出我最终使用阈值法处理的叶片。

![](RackMultipart20231001-1-okzepl_html_9549be1b05708a41.png)

阈值分割-病斑界面

还有一些预设的功能。

![](RackMultipart20231001-1-okzepl_html_e75e96c1288028ac.png)

形态学操作界面

其中形态学操作中只能对二值图像使用。掩膜功能除了当前窗口，还需要选中另一张图片。

![](RackMultipart20231001-1-okzepl_html_19af95a5e15aa6f5.png)

Kmeans方法界面

Kmeans中，只对RGB图像有效，伪彩色聚簇如上图所示，提取病斑将会自动提取预设的黑色像素。（这一功能换一张图片不一定会生效，原因是聚簇颜色不一定为黑色）

具体功能包括上述的大部分的操作内容。按Ctrl+Z可以回退到上一次的图像操作情况。

由于操作内容较多，此处省略一些具体的功能介绍。


