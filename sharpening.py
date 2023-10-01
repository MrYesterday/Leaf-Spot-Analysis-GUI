from EdgeDetection import roberts_edge_detection,sobel_edge_detection,prewitt_edge_detection,\
    laplacian_edge_detection,laplacian_of_gaussian_edge_detection
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 创建文件夹
if not os.path.exists('sharpening'):
    os.makedirs('sharpening')

# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')
# 分离RGB三个通道
b, g, r = cv2.split(img)

# 定义行数和列数
rows = 2
cols = 3

# 创建一个大小为 (15,8) 的窗口，用来显示多个子图
fig, ax = plt.subplots(rows, cols, figsize=(15, 8))


# 对每个通道应用边缘检测算法得到边缘图像
edge_b = roberts_edge_detection(b)
edge_g = roberts_edge_detection(g)
edge_r = roberts_edge_detection(r)

# 将边缘图像应用到每个通道上
alpha = 0.5 # 控制锐化程度的参数
enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)

# 将增强后的三个通道合并成一张彩色图像
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
cv2.imwrite('sharpening/enhanced_img_roberts.jpg', enhanced_img)
# 将BGR转换为RGB格式
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
ax[0, 0].imshow(enhanced_img)
ax[0, 0].set_title('Roberts Edge Sharpening')
ax[0, 0].axis('off')

# 对每个通道应用边缘检测算法得到边缘图像
edge_b = sobel_edge_detection(b)
edge_g = sobel_edge_detection(g)
edge_r = sobel_edge_detection(r)

# 将边缘图像应用到每个通道上
alpha = 0.5 # 控制锐化程度的参数
enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)

# 将增强后的三个通道合并成一张彩色图像
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
cv2.imwrite('sharpening/enhanced_img_sobel.jpg', enhanced_img)
# 将BGR转换为RGB格式
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
ax[0, 1].imshow(enhanced_img)
ax[0, 1].set_title('Sobel Edge Sharpening')
ax[0, 1].axis('off')

# 对每个通道应用边缘检测算法得到边缘图像
edge_b = prewitt_edge_detection(b)
edge_g = prewitt_edge_detection(g)
edge_r = prewitt_edge_detection(r)

# 将边缘图像应用到每个通道上
alpha = 0.5 # 控制锐化程度的参数
enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)

# 将增强后的三个通道合并成一张彩色图像
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
cv2.imwrite('sharpening/enhanced_img_prewitt.jpg', enhanced_img)
# 将BGR转换为RGB格式
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
ax[0, 2].imshow(enhanced_img)
ax[0, 2].set_title('prewitt Edge Sharpening')
ax[0, 2].axis('off')

# 对每个通道应用边缘检测算法得到边缘图像
edge_b = laplacian_edge_detection(b)
edge_g = laplacian_edge_detection(g)
edge_r = laplacian_edge_detection(r)

# 将边缘图像应用到每个通道上
alpha = 0.5 # 控制锐化程度的参数
enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)

# 将增强后的三个通道合并成一张彩色图像
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
cv2.imwrite('sharpening/enhanced_img_laplacian.jpg', enhanced_img)
# 将BGR转换为RGB格式
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
ax[1, 0].imshow(enhanced_img)
ax[1, 0].set_title('laplacian Edge Sharpening')
ax[1, 0].axis('off')

# 对每个通道应用边缘检测算法得到边缘图像
edge_b = laplacian_of_gaussian_edge_detection(b)
edge_g = laplacian_of_gaussian_edge_detection(g)
edge_r = laplacian_of_gaussian_edge_detection(r)

# 将边缘图像应用到每个通道上
alpha = 0.5 # 控制锐化程度的参数
enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)

# 将增强后的三个通道合并成一张彩色图像
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
cv2.imwrite('sharpening/enhanced_img_LoG.jpg', enhanced_img)
# 将BGR转换为RGB格式
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
ax[1, 1].imshow(enhanced_img)
ax[1, 1].set_title('LoG Edge Sharpening')
ax[1, 1].axis('off')

from DFT import GaussianHighFilter

# 对每个通道应用边缘检测算法得到边缘图像
edge_b = GaussianHighFilter(b,80)
edge_g = GaussianHighFilter(g,80)
edge_r = GaussianHighFilter(r,80)
edge_b = cv2.convertScaleAbs(edge_b)
edge_g = cv2.convertScaleAbs(edge_g)
edge_r = cv2.convertScaleAbs(edge_r)
# 将边缘图像应用到每个通道上
alpha = 0.5 # 控制锐化程度的参数
enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)

# 将增强后的三个通道合并成一张彩色图像
enhanced_img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
cv2.imwrite('sharpening/enhanced_img_GaussianHighFilter.jpg', enhanced_img)
# 将BGR转换为RGB格式
enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
ax[1, 2].imshow(enhanced_img)
ax[1, 2].set_title('GaussianHighFilter Edge Sharpening')
ax[1, 2].axis('off')
# 显示图像
plt.tight_layout()
plt.show()