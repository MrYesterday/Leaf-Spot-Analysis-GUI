from skimage.feature import graycomatrix, graycoprops

from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt


# 定义函数计算纹理特征
def compute_texture_features(image_path):
    # 读取图像
    img = io.imread(image_path)
    # 转换成灰度图像
    gray = color.rgb2gray(img)

    # 计算灰度共生矩阵
    glcm = graycomatrix(np.uint8(gray * 255), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    # 计算能量、对比度、相关性
    contrast = graycoprops(glcm, 'contrast')  # 对比度
    dissimilarity = graycoprops(glcm, 'dissimilarity')  # 差异性
    homogeneity = graycoprops(glcm, 'homogeneity')  # 均匀度
    energy = graycoprops(glcm, 'energy')  # 能量/角二阶矩
    correlation = graycoprops(glcm, 'correlation')  # 相关性

    return contrast[0, 0], dissimilarity[0, 0], homogeneity[0, 0], energy[0, 0], correlation[0, 0]


# 定义函数绘制五边形图
def plot_pentagon(features, feature_names):
    fig = plt.figure(figsize=(8, 7))
    angles = np.linspace(0, 2*np.pi, len(feature_names) + 1, endpoint=True)
    ax = fig.add_subplot(111, polar=True)
    plt.xticks(angles[:-1], feature_names)
    ax.plot(angles, [1]*len(angles), 'k--')
    for i, f in enumerate(features):
        values = np.array(f[:5])
        min_val = min(values)
        max_val = max(values)
        values = (values - min_val) / (max_val - min_val)
        ax.set_ylim()
        ax.plot(angles[:-1], values, linewidth=2, label="Image "+str(i+1))
        ax.fill(angles[:-1], values, alpha=0.25)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.show()



# 计算所有图像的纹理特征
features = []
for i in range(1, 11):
    image_path = 'AppleDiseaseLeaves/{}.jpg'.format(i)
    texture_features = compute_texture_features(image_path)
    features.append(texture_features)

# 绘制五边形图
feature_names = ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation']
for i in features:
    print(i)
plot_pentagon(features, feature_names)
