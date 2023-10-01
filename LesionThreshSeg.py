import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

if not os.path.exists('LesionThreshSeg'):
    os.makedirs('LesionThreshSeg')
# 读取图像
img = cv2.imread('RGB/r_channel.jpg')
# 读入叶片分割二值图像
mask = cv2.imread('ThresholdSegmentation/SegmentationOperation.jpg')
# 取反
mask = cv2.bitwise_not(mask)

# 对图像进行掩膜处理
masked_img = cv2.bitwise_and(img, mask)

# 显示掩膜处理后的图像和直方图

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 7))
ax[0,0].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
ax[0,0].axis('off')
ax[0,0].set_title('Mask processing')


# 显示直方图
ax[0,1].hist(masked_img.ravel(), bins=256, range=(1, 256))
ax[0,1].set_title('Histogram')

# 读取图像
leaf = cv2.imread('AppleDiseaseLeaves/1.jpg')
ax[0,2].imshow(cv2.cvtColor(cv2.bitwise_and(leaf, mask), cv2.COLOR_BGR2RGB))
ax[0,2].axis('off')
ax[0,2].set_title('Origin')
cv2.imwrite('LesionThreshSeg/Origin.jpg', cv2.bitwise_and(leaf, mask))

# 设定阈值
threshold_value = 99

# 阈值分割
ret, thresh = cv2.threshold(masked_img, threshold_value, 255, cv2.THRESH_BINARY)

# 显示分割后二值化图像
ax[1,0].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
ax[1,0].axis('off')
ax[1,0].set_title('thresh process')
cv2.imwrite('LesionThreshSeg/threshprocess.jpg', thresh)
'''
开运算消除大部分叶脉
'''
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,15))
# 对图像进行开运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
'''
开运算消除剩余噪点
'''
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# 对图像进行开运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imwrite('LesionThreshSeg/open operation.jpg', thresh)

# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
# 对图像进行闭运算
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# 显示分割后二值化图像
ax[1,1].imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
ax[1,1].axis('off')
ax[1,1].set_title('open and close operation')
cv2.imwrite('LesionThreshSeg/open and close operation.jpg', thresh)

tmp = np.uint8(cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY))
# 获取轮廓
contours, hierarchy = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
cv2.imwrite('LesionThreshSeg/contour.jpg', third_img)

# 对图像进行掩膜处理
Disease_img = cv2.bitwise_and(leaf, thresh)

# 显示分割后叶片病斑图像
ax[1,2].imshow(cv2.cvtColor(Disease_img, cv2.COLOR_BGR2RGB))
ax[1,2].axis('off')
ax[1,2].set_title('Disease Seg')
cv2.imwrite('LesionThreshSeg/Disease Seg.jpg', Disease_img)
plt.tight_layout()
plt.show()