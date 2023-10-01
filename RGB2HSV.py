import cv2
import matplotlib.pyplot as plt
import os

# 创建文件夹
if not os.path.exists('HSV'):
    os.makedirs('HSV')

# 读取图像
img = cv2.imread('sharpening/enhanced_img_sobel.jpg')

# 将图像从RGB空间转换为HSV空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 提取H、S、V三个通道的图像
h, s, v = cv2.split(hsv)

# 导出H通道图像
cv2.imwrite('HSV/h_channel.jpg', h)

# 导出S通道图像
cv2.imwrite('HSV/s_channel.jpg', s)

# 导出V通道图像
cv2.imwrite('HSV/v_channel.jpg', v)

# 设置直方图参数
hist_bins = 256
hist_range = (0, 256)

# 计算H、S、V三个通道的直方图
hist_h = cv2.calcHist([h], [0], None, [hist_bins], hist_range)
hist_s = cv2.calcHist([s], [0], None, [hist_bins], hist_range)
hist_v = cv2.calcHist([v], [0], None, [hist_bins], hist_range)

# 在plt窗口显示H、S、V三个通道的直方图
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
ax1.hist(h.ravel(), bins=hist_bins, range=hist_range)
ax1.set_title('Hue Histogram')
ax2.hist(s.ravel(), bins=hist_bins, range=hist_range)
ax2.set_title('Saturation Histogram')
ax3.hist(v.ravel(), bins=hist_bins, range=hist_range)
ax3.set_title('Value Histogram')
plt.show()
