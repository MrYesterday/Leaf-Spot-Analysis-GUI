import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')

# 计算缩放比例
scale_percent = 800 / img.shape[1]
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
# 将图像转换为灰度图
gray = cv2.cvtColor(cv2.resize(img, (width, height)), cv2.COLOR_BGR2GRAY)

# 绘制灰度图

cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 绘制灰度直方图
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('Pixels')
plt.show()
