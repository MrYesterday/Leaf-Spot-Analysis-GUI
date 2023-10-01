import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
# 创建文件夹
if not os.path.exists('ThresholdSegmentation'):
    os.makedirs('ThresholdSegmentation')
# 读取图像
img = cv2.imread('sharpening/enhanced_img_sobel.jpg')


def THRESH_OTSU():
    # 转换为HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 分离通道
    h, s, v = cv2.split(hsv)
    r, g, b = cv2.split(img)

    # OTSU阈值法
    otsu_thresh = cv2.THRESH_OTSU

    # 分别对每个通道进行二值化处理
    _, h_th = cv2.threshold(h, 0, 255, otsu_thresh)
    _, s_th = cv2.threshold(s, 0, 255, otsu_thresh)
    _, v_th = cv2.threshold(v, 0, 255, otsu_thresh)
    _, r_th = cv2.threshold(r, 0, 255, otsu_thresh)
    _, g_th = cv2.threshold(g, 0, 255, otsu_thresh)
    _, b_th = cv2.threshold(b, 0, 255, otsu_thresh)

    # 显示二值化结果
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    axes[0, 0].imshow(h_th, cmap='gray')
    axes[0, 0].set_title('Hue Thresholding')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(s_th, cmap='gray')
    axes[0, 1].set_title('Saturation Thresholding')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(v_th, cmap='gray')
    axes[0, 2].set_title('Value Thresholding')
    axes[0, 2].axis('off')
    axes[1, 0].imshow(r_th, cmap='gray')
    axes[1, 0].set_title('Red Thresholding')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(g_th, cmap='gray')
    axes[1, 1].set_title('Green Thresholding')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(b_th, cmap='gray')
    axes[1, 2].set_title('Blue Thresholding')
    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.show()

# THRESH_OTSU()

'''
双阈值法，分割效果不好，完整叶片无法分离
'''
def DoubleThresholdSegmentation():
    '''
    双阈值法分割
    '''
    # 读取图像并转换为灰度图像
    img = cv2.imread('smoothing/median.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算梯度图像
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad = cv2.sqrt(cv2.addWeighted(cv2.pow(sobel_x, 2.0), 1.0, cv2.pow(sobel_y, 2.0), 1.0, 0.0))

    # 计算直方图
    hist, bins = np.histogram(grad.ravel(), 256, [0, 256])

    # 自适应计算阈值
    total_pixels = gray.shape[0] * gray.shape[1]
    sum_ = 0.0
    for i in range(256):
        sum_ += i * hist[i]
        w0 = float(np.sum(hist[:i + 1])) / total_pixels
        w1 = 1.0 - w0
        if w0 == 0.0 or w1 == 0.0:
            continue
        mean0 = float(np.sum(hist[:i + 1] * np.arange(i + 1))) / float(np.sum(hist[:i + 1]))
        mean1 = float(sum_ - np.sum(hist[:i + 1] * np.arange(i + 1))) / float(total_pixels - np.sum(hist[:i + 1]))
        var_between = w0 * w1 * (mean0 - mean1) ** 2
        if i == 0 or var_between > max_var_between:
            max_var_between = var_between
            threshold = i

    # 标记像素
    low = threshold / 2.0
    high = threshold * 1.5
    mask_low = grad < low
    mask_high = grad > high
    mask_middle = np.logical_and(grad >= low, grad <= high)
    result = np.zeros(gray.shape, dtype=np.uint8)
    result[mask_low] = 0
    result[mask_high] = 255
    result[mask_middle] = 0

    # 可视化分割结果
    plt.imshow(result, cmap='gray')
    plt.show()


# DoubleThresholdSegmentation()

'''
多通道阈值分割并叠加，噪点开运算消除解决！效果较好，分离的叶片比较完整
'''
# 转换为HSV空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 分离通道
h, s, v = cv2.split(hsv)
r, g, b = cv2.split(img)
otsu_thresh = cv2.THRESH_OTSU
_, s_th = cv2.threshold(s, 0, 255, otsu_thresh)
_, r_th = cv2.threshold(r, 0, 255, otsu_thresh)

# 取反
s_th = cv2.bitwise_not(s_th)
# 叠加
result = cv2.bitwise_or(r_th, s_th)
cv2.imwrite('ThresholdSegmentation/Origin.jpg', result)
# # 显示结果
# plt.imshow(result, cmap='gray')
# plt.axis('off')
# plt.show()
# 定义结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))


# 显示二值化结果
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 6))
axes[0].imshow(result, cmap='gray')
axes[0].axis('off')
axes[0].set_title('Origin')
# 进行开运算操作
result_open = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
# 显示结果
axes[1].imshow(result_open, cmap='gray')
axes[1].axis('off')
axes[1].set_title('First Open operation')

# 进行闭运算操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
closed = cv2.morphologyEx(result_open, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('ThresholdSegmentation/SegmentationOperation.jpg', closed)
# 显示结果
axes[2].imshow(closed, cmap='gray')
axes[2].axis('off')
axes[2].set_title('Second Close operation')

'''
图像Mask进行分割
'''
Mask = cv2.bitwise_not(closed)
# 将mask三通道扩展为与img相同的通道数
MergeMask = cv2.merge([Mask, Mask, Mask])
# 将img与mask相乘，保留mask为255的像素，其他像素置为0
masked_img = cv2.bitwise_and(img, MergeMask)
cv2.imwrite('ThresholdSegmentation/LeafSegmentation.jpg', masked_img)
# 显示结果
axes[3].imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
axes[3].axis('off')
axes[3].set_title('Segmentation')
plt.tight_layout()
plt.show()
