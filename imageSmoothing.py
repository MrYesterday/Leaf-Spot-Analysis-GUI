import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 创建文件夹
if not os.path.exists('smoothing'):
    os.makedirs('smoothing')
# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')

# 定义卷积核大小
kernel_size = 5

# 高斯滤波器
img_gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# 中值滤波器
img_median = cv2.medianBlur(img, kernel_size)

# 均值滤波器
img_mean = cv2.blur(img, (kernel_size, kernel_size))

cv2.imwrite('smoothing/gaussian.jpg', img_gaussian)
cv2.imwrite('smoothing/median.jpg', img_median)
cv2.imwrite('smoothing/mean.jpg', img_mean)

# 显示原始图像和处理后的图像
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(img_gaussian, cv2.COLOR_BGR2RGB))
plt.title('Gaussian')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(img_median, cv2.COLOR_BGR2RGB))
plt.title('Median')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img_mean, cv2.COLOR_BGR2RGB))
plt.title('Mean')
plt.axis('off')
plt.show()
