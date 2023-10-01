import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# 创建文件夹
if not os.path.exists('cannySeg'):
    os.makedirs('cannySeg')
'''
直接使用canny算子 出现边缘断裂/内部噪点边的问题 调整阈值效果也不佳 +高斯平滑后效果好些 但是仍然有噪点和杂边 边缘出现小断裂
'''
# 读取图像
img = cv2.imread('sharpening/enhanced_img_sobel.jpg')
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7))
# 转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用高斯滤波器平滑图像
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 使用Canny算子进行边缘检测
edges = cv2.Canny(blur, 80, 150)
cv2.imwrite('cannySeg/CannyDetect.jpg', edges)
ax[0, 0].imshow(edges, cmap='gray')
ax[0, 0].axis('off')
ax[0, 0].set_title('Canny edge Detect')
# 进行形态学膨胀处理
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
dilated = cv2.dilate(edges, kernel, iterations=2)

# 显示结果
cv2.imwrite('cannySeg/dilated.jpg', dilated)
ax[0, 1].imshow(dilated, cmap='gray')
ax[0, 1].axis('off')
ax[0, 1].set_title('Morphological dilate processing')
# 找到边缘所围成的轮廓
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建空图像，大小与输入图像相同
mask = np.zeros_like(img)

# 填充轮廓
cv2.drawContours(mask, contours, -1, (255,255,255), cv2.FILLED)

# 定义腐蚀结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
# 进行腐蚀处理
mask = cv2.erode(mask, kernel, iterations=2)

cv2.imwrite('cannySeg/mask.jpg', mask)
# 显示结果
ax[1, 0].imshow(mask)
ax[1, 0].axis('off')
ax[1, 0].set_title('filled and erode')

cv2.imwrite('cannySeg/result.jpg', cv2.bitwise_and(mask, img))
result = cv2.cvtColor(cv2.bitwise_and(mask, img), cv2.COLOR_BGR2RGB)
# 显示结果
ax[1,1].imshow(result)
ax[1,1].axis('off')
ax[1, 1].set_title('canny seg result')
plt.tight_layout()
plt.show()



