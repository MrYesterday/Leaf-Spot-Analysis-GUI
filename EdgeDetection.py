import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def roberts_edge_detection(img):
    # 定义Roberts算子
    roberts_x = np.array([[-1, 0], [0, 1]])
    roberts_y = np.array([[0, -1], [1, 0]])

    # 对图像进行卷积操作
    grad_x = cv2.filter2D(img, -1, roberts_x)
    grad_y = cv2.filter2D(img, -1, roberts_y)

    # 计算梯度幅值
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 返回边缘检测结果
    return grad.astype(np.uint8)


def sobel_edge_detection(img, ksize=3):
    # 使用Sobel算子进行边缘检测
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    # 计算梯度幅值
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 返回边缘检测结果
    return grad.astype(np.uint8)


def prewitt_edge_detection(img):
    # 使用Prewitt算子进行边缘检测
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)

    # 计算梯度幅值
    grad = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 返回梯度幅值
    return grad.astype(np.uint8)


def laplacian_edge_detection(img, ksize=3):
    # 使用Laplacian算子进行边缘检测
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)

    # 计算梯度幅值
    grad = np.abs(laplacian)

    # 返回梯度幅值
    return grad.astype(np.uint8)


def laplacian_of_gaussian_edge_detection(img, ksize=3, sigma=1.0):
    """
    使用LoG算子进行边缘检测，得到边缘锐化后的图像
    :param img: 原始图像，灰度图
    :param ksize: 模糊核大小，必须为正奇数
    :param sigma: LoG算子的高斯模糊参数
    :return: 边缘锐化后的图像
    """
    # 使用高斯模糊对图像进行平滑处理
    img_smooth = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    # 使用Laplacian算子进行边缘检测
    laplacian = cv2.Laplacian(img_smooth, cv2.CV_64F, ksize=ksize)

    # 计算梯度幅值
    grad = np.abs(laplacian)

    # 返回梯度幅值
    return grad.astype(np.uint8)

if __name__ == "__main__":
    # 创建文件夹
    if not os.path.exists('EdgeDetection'):
        os.makedirs('EdgeDetection')

    # 读取图片
    img = cv2.imread('AppleDiseaseLeaves/1.jpg', 0)  # 以灰度模式读取图片

    # # 应用Roberts算子进行边缘锐化
    # edge_detection_result = roberts_edge_detection(img)
    #
    # # 显示图像
    # plt.imshow(edge_detection_result, cmap='gray')
    # plt.title('Roberts Edge Detection')
    # plt.axis('off')
    # plt.show()

    # # 应用Sobel算子进行边缘检测
    # edge_detection_result = sobel_edge_detection(img, ksize=3)
    #
    # # 显示图像
    # plt.imshow(edge_detection_result, cmap='gray')
    # plt.title('Sobel Edge Detection')
    # plt.axis('off')
    # plt.show()

    # # 应用prewitt算子进行边缘检测
    # edge_detection_result = prewitt_edge_detection(img)
    #
    # # 显示图像
    # plt.imshow(edge_detection_result, cmap='gray')
    # plt.title('Prewitt Edge Detection')
    # plt.axis('off')
    # plt.show()

    # # 应用prewitt算子进行边缘检测
    # edge_detection_result = laplacian_edge_detection(img, ksize=3)
    #
    # # 显示图像
    # plt.imshow(edge_detection_result, cmap='gray')
    # plt.title('laplacian Edge Detection')
    # plt.axis('off')
    # plt.show()

    # # 应用LoG算子进行边缘锐化
    # edge_detection_result = laplacian_of_gaussian_edge_detection(img, ksize=3)
    #
    # # 显示图像
    # plt.imshow(edge_detection_result, cmap='gray')
    # plt.title('LoG Edge Detection')
    # plt.axis('off')
    # plt.show()

    # 定义行数和列数
    rows = 2
    cols = 3

    # 创建一个大小为 (15,8) 的窗口，用来显示多个子图
    fig, ax = plt.subplots(rows, cols, figsize=(15, 8))

    # 应用Roberts算子进行边缘锐化
    edge_detection_result = roberts_edge_detection(img)
    plt.imsave('EdgeDetection/RobertsEdgeDetection.png', edge_detection_result, cmap='gray')
    ax[0, 0].imshow(edge_detection_result, cmap='gray')
    ax[0, 0].set_title('Roberts Edge Detection')
    ax[0, 0].axis('off')

    # 应用Sobel算子进行边缘检测
    edge_detection_result = sobel_edge_detection(img, ksize=3)
    plt.imsave('EdgeDetection/SobelEdgeDetection.png', edge_detection_result, cmap='gray')
    ax[0, 1].imshow(edge_detection_result, cmap='gray')
    ax[0, 1].set_title('Sobel Edge Detection')
    ax[0, 1].axis('off')

    # 应用prewitt算子进行边缘检测
    edge_detection_result = prewitt_edge_detection(img)
    plt.imsave('EdgeDetection/PrewittEdgeDetection.png', edge_detection_result, cmap='gray')
    ax[0, 2].imshow(edge_detection_result, cmap='gray')
    ax[0, 2].set_title('Prewitt Edge Detection')
    ax[0, 2].axis('off')

    # 应用laplacian算子进行边缘检测
    edge_detection_result = laplacian_edge_detection(img, ksize=3)
    plt.imsave('EdgeDetection/LaplacianEdgeDetection.png', edge_detection_result, cmap='gray')
    ax[1, 0].imshow(edge_detection_result, cmap='gray')
    ax[1, 0].set_title('Laplacian Edge Detection')
    ax[1, 0].axis('off')

    # 应用LoG算子进行边缘锐化
    edge_detection_result = laplacian_of_gaussian_edge_detection(img, ksize=3)
    plt.imsave('EdgeDetection/LoGEdgeDetection.png', edge_detection_result, cmap='gray')
    ax[1, 1].imshow(edge_detection_result, cmap='gray')
    ax[1, 1].set_title('LoG Edge Detection')
    ax[1, 1].axis('off')

    ax[1, 2].axis('off')

    # 显示图像
    plt.show()
