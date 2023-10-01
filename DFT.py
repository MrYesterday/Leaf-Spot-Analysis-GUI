import cv2
import numpy as np
import matplotlib.pyplot as plt


# ideal lowpass filter
def lowPassFilter(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                if dis <= d:
                    transfor_matrix[i, j] = 1
                else:
                    transfor_matrix[i, j] = 0
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


# gaussian lowpass filter
def GaussianLowFilter(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transfor_matrix[i, j] = np.exp(-(dis ** 2) / (2 * (d ** 2)))
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


# ideal highpass filter
def highPassFilter(image, d):  # D为阈值
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                if dis <= d:
                    transfor_matrix[i, j] = 0
                else:
                    transfor_matrix[i, j] = 1
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


# gaussian highpass
def GaussianHighFilter(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, image.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                transfor_matrix[i, j] = 1 - np.exp(-(dis ** 2) / (2 * (d ** 2)))
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img
if __name__ == "__main__":
    import os

    if not os.path.exists('DFT'):
        os.makedirs('DFT')

    img_man = cv2.imread('AppleDiseaseLeaves/1.jpg', 0)  # 直接读为灰度图像
    plt.subplot(221)
    plt.imshow(img_man, 'gray')
    plt.title('origial')
    plt.xticks([])
    plt.yticks([])

    out = GaussianLowFilter(img_man, 100)
    cv2.imwrite('DFT/GaussianLowFilter.jpg', out)
    plt.subplot(222)
    plt.imshow(out, 'gray')
    plt.title('Lowpass')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(223)
    plt.imshow(img_man, 'gray')
    plt.title('origial')
    plt.xticks([])
    plt.yticks([])

    out = GaussianHighFilter(img_man, 80)
    cv2.imwrite('DFT/GaussianHighFilter.jpg', out)
    plt.subplot(224)
    plt.imshow(out, 'gray')
    plt.title('Highpass')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    # img = cv2.imread('AppleDiseaseLeaves/1.jpg')
    # r, g, b = cv2.split(img)
    # smooth_b = GaussianLowFilter(b,100)
    # smooth_g = GaussianLowFilter(g,100)
    # smooth_r = GaussianLowFilter(r,100)
    # enhanced_img = cv2.merge((smooth_r, smooth_g, smooth_b))
    # cv2.imwrite('DFT/GaussianLowFilter.jpg',enhanced_img)