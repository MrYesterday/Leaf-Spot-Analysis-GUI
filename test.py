import cv2
from matplotlib import pyplot as plt

# 读取图片
img = cv2.imread('AppleDiseaseLeaves/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(gray.ravel(), bins=256, range=[0, 256])
plt.title('Gray Image Histogram')
plt.show()
