import cv2
import matplotlib.pyplot as plt
import numpy as np


# 读取图像
img = cv2.imread('RegionGrowing/RegionGrowingMask.jpg')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
# # 定义结构元素
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
# # 对图像进行开运算
# thresh = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# cv2.imwrite('RegionGrowing/SegOpenOP.jpg', thresh)
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
# 对图像进行闭运算
thresh = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('RegionGrowing/SegCloseOP.jpg', thresh)

# # 转换为灰度图像
# gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
# # 查找边缘
# contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# # 绘制填充后的图像
# filled_img = np.zeros_like(thresh)
# cv2.drawContours(filled_img, contours, -1, (255, 255, 255), thickness=-1)
# cv2.imwrite('RegionGrowing/SegFilled.jpg', filled_img)
#
# ax.imshow(filled_img)
#
# ax.axis('off')
# ax.set_title('morphology processing')
# plt.show()