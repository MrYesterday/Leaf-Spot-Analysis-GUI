import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# 创建文件夹
if not os.path.exists('RegionGrowing'):
    os.makedirs('RegionGrowing')
# 读取彩色图像并转换为灰度图像
# img = cv2.imread('sharpening/enhanced_img_sobel.jpg')
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# h, w = gray_img.shape[:2]

# 读取图像
img = cv2.imread('sharpening/enhanced_img_sobel.jpg')

# # 将图像从RGB空间转换为HSV空间
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# # 提取H、S、V三个通道的图像
# h, s, v = cv2.split(hsv)
#
# gray_img = v

# 提取H、S、V三个通道的图像
b,g,r = cv2.split(img)

gray_img = r

h, w = gray_img.shape[:2]


def get_binary_img(img):
    # gray img to bin image
    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 127 else 0
    return bin_img


# 调用
bin_img = get_binary_img(gray_img)
out_img = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
# 选择初始3个种子点
# seeds = [(887, 833), (1049, 989), (1373, 545)]
seeds = [(887, 833), (1049, 989)]
for seed in seeds:
    x = seed[0]
    y = seed[1]
    out_img[y][x] = 255

# 8 邻域
directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
visited = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
while len(seeds):
    seed = seeds.pop(0)
    x = seed[0]
    y = seed[1]
    # visit point (x,y)
    visited[y][x] = 1
    for direct in directs:
        cur_x = x + direct[0]
        cur_y = y + direct[1]
        # 非法
        if cur_x < 0 or cur_y < 0 or cur_x >= w or cur_y >= h:
            continue
        # 没有访问过且属于同一目标
        if (not visited[cur_y][cur_x]) and (bin_img[cur_y][cur_x] == bin_img[y][x]):
            out_img[cur_y][cur_x] = 255
            visited[cur_y][cur_x] = 1
            seeds.append((cur_x, cur_y))

# 定义行数和列数
rows = 1
cols = 2

# 创建一个大小为 (15,8) 的窗口，用来显示多个子图
fig, ax = plt.subplots(rows, cols, figsize=(15, 8))
cv2.imwrite('RegionGrowing/RegionGrowingMask.jpg', out_img)
ax[0].imshow(out_img,cmap='gray')
ax[0].set_title('mask')
ax[0].axis('off')

bake_img = cv2.imread('AppleDiseaseLeaves/1.jpg')
h = bake_img.shape[0]
w = bake_img.shape[1]
for i in range(h):
    for j in range(w):
        if out_img[i][j] == 255:
            bake_img[i][j][0] = 0
            bake_img[i][j][1] = 0
            bake_img[i][j][2] = 0

# 可视化分割结果
cv2.imwrite('RegionGrowing/RegionGrowingResult.jpg', bake_img)
bake_img = cv2.cvtColor(bake_img, cv2.COLOR_BGR2RGB)
ax[1].imshow(bake_img)
ax[1].set_title('object')
ax[1].axis('off')
plt.show()