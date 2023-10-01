import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 创建文件夹
if not os.path.exists('shape'):
    os.makedirs('shape')
# 读取图像
img = cv2.imread('ThresholdSegmentation/SegmentationOperation.jpg')
img = cv2.bitwise_not(img)
# 将图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化处理
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 获取轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 计算面积和周长
area = cv2.contourArea(contours[0])
perimeter = cv2.arcLength(contours[0], True)

# 计算矩形
x,y,w,h = cv2.boundingRect(contours[0])

# 计算圆形度
circularity = 4*np.pi*area/(perimeter**2)

# 计算重心
M = cv2.moments(contours[0])
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])


# 计算复杂度
e = (perimeter ** 2) / area

# 输出结果
print('面积:', area)
print('周长:', perimeter)
print('长度:', w)
print('宽度:', h)
print('圆形度:', circularity)
print('重心:', cx, cy)
print('复杂度:', e)
img = cv2.imread('AppleDiseaseLeaves/1.jpg')
first_img = img.copy()
# 绘制矩形和重心
cv2.rectangle(first_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.circle(first_img, (cx, cy), 5, (0, 0, 255), -1)

# 计算最小外接矩形
rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rect)
box = np.int0(box)

Second_img = img.copy()
# 计算最小外接矩形的面积
rect_area = cv2.contourArea(box)
print('最小外接矩形面积为:', rect_area)
print('矩形度:', area/rect_area)
# 在图像上绘制最小外接矩形
cv2.drawContours(Second_img, [box], 0, (0, 0, 255), 2)

# 计算最小外接矩形的重心
M = cv2.moments(box)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print("最小外接矩形重心坐标：({}, {})".format(cx, cy))

# 在图像上绘制重心
cv2.drawMarker(Second_img, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)

# 生成空白图片
third_img = np.zeros_like(img)
# 绘制轮廓
cv2.drawContours(third_img, contours, -1, (0, 255, 0), 2)

# 显示原始图像和绘制了矩形和重心的图像
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
ax[0,0].imshow(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB))
ax[0,0].axis('off')
ax[0,0].set_title('Original Image bounding rectangle')
ax[0,1].imshow(cv2.cvtColor(Second_img, cv2.COLOR_BGR2RGB))
ax[0,1].axis('off')
ax[0,1].set_title('Original Image Mini bounding rectangle')
ax[1,0].imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
ax[1,0].axis('off')
ax[1,0].set_title('binary image')
ax[1,1].imshow(third_img)
ax[1,1].axis('off')
ax[1,1].set_title('contour image')
plt.tight_layout()
plt.show()