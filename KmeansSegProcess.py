import cv2
import matplotlib.pyplot as plt
import numpy as np


# 读取图像
img = cv2.imread('RGBKmeansCluster/Seg.jpg')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,7))
# 对图像进行开运算
thresh = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imwrite('RGBKmeansCluster/SegOpenOP.jpg', thresh)
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,15))
# 对图像进行闭运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('RGBKmeansCluster/SegCloseOP.jpg', thresh)
# 转换为灰度图像
gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

# 查找边缘
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 计算所有连通分量的面积总和
total_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    total_area += area

print("连通分量的面积总和：", total_area)
print("病斑占比：", total_area/1051978.0)
# 生成空白图片
third_img = np.zeros_like(img)
# 绘制轮廓
cv2.drawContours(third_img, contours, -1, (0, 255, 0), 2)
cv2.imwrite('RGBKmeansCluster/contour.jpg', third_img)



# 绘制填充后的图像
filled_img = np.zeros_like(thresh)
cv2.drawContours(filled_img, contours, -1, (255, 255, 255), thickness=-1)
cv2.imwrite('RGBKmeansCluster/SegFilled.jpg', filled_img)

ax.imshow(filled_img)

ax.axis('off')
ax.set_title('morphology processing')
plt.show()