import cv2
from matplotlib import pyplot as plt
import os
if not os.path.exists('RGB'):
    os.makedirs('RGB')
# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')

# 分离通道
b, g, r = cv2.split(img)

# 导出H通道图像
cv2.imwrite('RGB/b_channel.jpg', b)

# 导出S通道图像
cv2.imwrite('RGB/g_channel.jpg', g)

# 导出V通道图像
cv2.imwrite('RGB/r_channel.jpg', r)

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
# plt.figure(figsize=(12, 5))

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


plt.tight_layout()
plt.show()