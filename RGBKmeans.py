import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('ThresholdSegmentation/LeafSegmentation.jpg')

# 转换为RGB颜色空间
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255],
                   [255, 255, 255], [0, 0, 0]], dtype=np.float32)

height, width = img.shape[:2]

k = 7
# 将像素点进行reshape，以方便使用聚类算法
img_reshaped = img.reshape(height * width, 3)

# 使用KMeans算法进行聚类，将图像分为k个类别
kmeans = KMeans(n_clusters=k, random_state=0).fit(img_reshaped)

# 根据聚类结果对图像进行重构
new_img = np.zeros_like(img_reshaped)
for i in range(k):
    new_img[kmeans.labels_ == i] = colors[i]
new_img = new_img.reshape(height, width, 3)
'''
# 将图像转换为一维数组
img_data = img.reshape((-1, 3))
# 设置聚类数量
k = 4
# 使用K均值聚类算法进行聚类
kmeans = KMeans(n_clusters=k)
kmeans.fit(img_data)

# 获取每个像素所属的类别
labels = kmeans.labels_

# 用每个类别的平均RGB值替换该类别中所有像素的值
# colors = kmeans.cluster_centers_.astype(np.uint8)
new_img_data = colors[labels]
new_img = new_img_data.reshape(img.shape)
'''
import os
if not os.path.exists('RGBKmeansCluster'):
    os.makedirs('RGBKmeansCluster')
cv2.imwrite('RGBKmeansCluster/result.jpg', cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
# 显示原始图像和聚类后的图像
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
ax[0].imshow(img)
ax[0].axis('off')
ax[0].set_title('Origin')
ax[1].imshow(new_img)
ax[1].axis('off')
ax[1].set_title('Clustered Image')
plt.tight_layout()
plt.show()


# 提取黑色像素
black_pixels = np.all(new_img == [0, 0, 0], axis=-1)
copy_img = np.zeros_like(new_img)
# 将黑色像素设置为白色，其他像素设置为黑色
copy_img[black_pixels] = [255, 255, 255]
copy_img[~black_pixels] = [0, 0, 0]

cv2.imwrite('RGBKmeansCluster/Seg.jpg', copy_img)
# 显示结果
plt.imshow(cv2.cvtColor(copy_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()