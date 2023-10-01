import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox,  Label
from tkinter.ttk import Combobox
import cv2
import numpy as np
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import ctypes

# 设置应用程序栏图标
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("icon.ico")
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

def THRESH_OTSU(img):
    if not is_rgb_image(img):
        messagebox.showerror("error", "大津法只接受RGB图像")
        return
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

def DoubleThresholdSegmentation(img):
    '''
    双阈值法分割
    '''
    # 读取图像并转换为灰度图像
    if not is_rgb_image(img):
        messagebox.showerror("error", "双阈值法只接受RGB图像")
        return
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

    return result

# 创建主窗口
root = tk.Tk()
root.title("图像处理程序")
root.geometry("1200x800")
root.resizable(False, False)
root.iconbitmap("./icon.ico")
from keyboard import add_hotkey,  unhook_all
from threading import Thread

def ctrl_z_handler():
    global img, prev_img
    img = prev_img.copy()
    show_image()


# 全局变量
img = None
prev_img = None
origin = None
def is_rgb_image(image):
    # 检查通道数
    channel_count = image.shape[2] if len(image.shape) == 3 else 1

    if channel_count == 3:
        return True
    else:
        return False

def is_binary_image(image):
    if is_rgb_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将图像二值化
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 统计图像中非零像素的数量
    non_zero_pixels = cv2.countNonZero(binary_image)

    # 判断是否为二值图像
    is_binary = non_zero_pixels == 0 or non_zero_pixels == image.size

    return is_binary

# 打开图片
def open_image():
    global img, prev_img, origin
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        origin = file_path
        img = cv2.imread(file_path)
        prev_img = img.copy()
        show_image()

        # 启用保存图片和重置图片菜单项
        save_menu.entryconfig(0, state=tk.NORMAL)
        reset_menu.entryconfig(0, state=tk.NORMAL)

# 保存图片
def save_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
    if file_path:
        cv2.imwrite(file_path, img)

# 重置图片
def reset_image():
    global img, prev_img, origin
    img = cv2.imread(origin)
    prev_img = img.copy()
    show_image()

# 灰度化处理
def convert_to_gray():
    global img, prev_img
    prev_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image()

# 灰度直方图
def show_histogram():
    if is_rgb_image(img):
        tmp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        tmp_img = img.copy()
    hist = cv2.calcHist([tmp_img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.show()

# 图像通道处理
def channel_handler(channel, hist=False):
    global img, prev_img
    prev_img = img.copy()
    # 分离三通道
    if not is_rgb_image(img):
        messagebox.showerror(title="error", message="错误, 当前图片不是RGB图像。")
        img = prev_img.copy()
        return
    if channel == 'R':
        if hist:
            b, g, r = cv2.split(img)
            plt.hist(b.ravel(), bins=256, color='b', range=[0, 256])
            plt.title('Blue Channel Histogram')
            plt.show()
        else:
            img = img[:, :, 0]
    elif channel == 'G':
        if hist:
            b, g, r = cv2.split(img)
            plt.hist(g.ravel(), bins=256, color='g', range=[0, 256])
            plt.title('Green Channel Histogram')
            plt.show()
        else:
            img = img[:, :, 1]
    elif channel == 'B':
        if hist:
            b, g, r = cv2.split(img)
            plt.hist(r.ravel(), bins=256, color='r', range=[0, 256])
            plt.title('Red Channel Histogram')
            plt.show()
        else:
            img = img[:, :, 2]
    elif channel == 'H':
        if hist:
            # 将图像从RGB空间转换为HSV空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 提取H、S、V三个通道的图像
            h, s, v = cv2.split(hsv)
            plt.hist(h.ravel(), bins=256, color='b', range=[0, 256])
            plt.title('H Channel Histogram')
            plt.show()
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 0]
    elif channel == 'S':
        if hist:
            # 将图像从RGB空间转换为HSV空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 提取H、S、V三个通道的图像
            h, s, v = cv2.split(hsv)
            plt.hist(s.ravel(), bins=256, color='g', range=[0, 256])
            plt.title('S Channel Histogram')
            plt.show()
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]
    elif channel == 'V':
        if hist:
            # 将图像从RGB空间转换为HSV空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 提取H、S、V三个通道的图像
            h, s, v = cv2.split(hsv)
            plt.hist(v.ravel(), bins=256, color='b', range=[0, 256])
            plt.title('V Channel Histogram')
            plt.show()
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]

    show_image()

# 创建加载中提示窗口
def show_loading_window():
    loading_window = tk.Toplevel()
    loading_window.title("Loading")
    loading_window.geometry("200x100")
    loading_label = Label(loading_window, text="Processing...", font=("Helvetica", 12))
    loading_label.pack(pady=20)

    return loading_window

# 平滑处理
def smooth_handler(method, thresh=100):
    global img, prev_img
    prev_img = img.copy()

    # 创建并显示加载中提示窗口
    loading_window = show_loading_window()

    def process_smooth():
        global img
        nonlocal loading_window
        if method == 'Spatial Gaussian':
            img = cv2.GaussianBlur(img, (5, 5), 0)
        elif method == 'Mean':
            img = cv2.blur(img, (5, 5))
        elif method == 'Median':
            img = cv2.medianBlur(img, 5)
        elif method == 'Frequency Gaussian':
            if is_rgb_image(img):
                b, g, r = cv2.split(img)
                smooth_b = GaussianLowFilter(b, thresh)
                smooth_g = GaussianLowFilter(g, thresh)
                smooth_r = GaussianLowFilter(r, thresh)
                img = cv2.merge((smooth_b, smooth_g, smooth_r))
            else:
                img = GaussianLowFilter(img, thresh)
            img = cv2.convertScaleAbs(img)

        # 更新图像显示
        show_image()

        # 关闭加载中提示窗口
        loading_window.destroy()

    # 使用线程来执行平滑处理
    Thread(target=process_smooth).start()
# 显示图片
def show_image():
    scale = min(800 / img.shape[0], 1200 / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
    image_label.configure(image=img_tk)
    image_label.image = img_tk


def sharpen_handler(param, thresh=100):
    global img, prev_img
    prev_img = img.copy()

    # 创建并显示加载中提示窗口
    loading_window = show_loading_window()

    def process_sharpen():
        global img
        nonlocal loading_window
        if param == "Roberts":
            # 处理Roberts算子锐化逻辑
            if not is_rgb_image(img):
                edge = roberts_edge_detection(img)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                img = cv2.addWeighted(img, 1, edge, alpha, 0)
            else:
                # 分离RGB三个通道
                b, g, r = cv2.split(img)
                # 对每个通道应用边缘检测算法得到边缘图像
                edge_b = roberts_edge_detection(b)
                edge_g = roberts_edge_detection(g)
                edge_r = roberts_edge_detection(r)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
                enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
                enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)
                # 将增强后的三个通道合并成一张彩色图像
                img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
        elif param == "Sobel":
            # 处理Sobel算子锐化逻辑
            if not is_rgb_image(img):
                edge = sobel_edge_detection(img)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                img = cv2.addWeighted(img, 1, edge, alpha, 0)
            else:
                # 分离RGB三个通道
                b, g, r = cv2.split(img)
                # 对每个通道应用边缘检测算法得到边缘图像
                edge_b = sobel_edge_detection(b)
                edge_g = sobel_edge_detection(g)
                edge_r = sobel_edge_detection(r)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
                enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
                enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)
                # 将增强后的三个通道合并成一张彩色图像
                img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
        elif param == "Prewitt":
            # 处理Prewitt算子锐化逻辑
            if not is_rgb_image(img):
                edge = prewitt_edge_detection(img)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                img = cv2.addWeighted(img, 1, edge, alpha, 0)
            else:
                # 分离RGB三个通道
                b, g, r = cv2.split(img)
                # 对每个通道应用边缘检测算法得到边缘图像
                edge_b = prewitt_edge_detection(b)
                edge_g = prewitt_edge_detection(g)
                edge_r = prewitt_edge_detection(r)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
                enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
                enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)
                # 将增强后的三个通道合并成一张彩色图像
                img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
        elif param == "Laplacian":
            # 处理Laplacian算子锐化逻辑
            if not is_rgb_image(img):
                edge = laplacian_edge_detection(img)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                img = cv2.addWeighted(img, 1, edge, alpha, 0)
            else:
                # 分离RGB三个通道
                b, g, r = cv2.split(img)
                # 对每个通道应用边缘检测算法得到边缘图像
                edge_b = laplacian_edge_detection(b)
                edge_g = laplacian_edge_detection(g)
                edge_r = laplacian_edge_detection(r)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
                enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
                enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)
                # 将增强后的三个通道合并成一张彩色图像
                img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
        elif param == "LoG":
            # 处理LoG算子锐化逻辑
            if not is_rgb_image(img):
                edge = laplacian_of_gaussian_edge_detection(img)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                img = cv2.addWeighted(img, 1, edge, alpha, 0)
            else:
                # 分离RGB三个通道
                b, g, r = cv2.split(img)
                # 对每个通道应用边缘检测算法得到边缘图像
                edge_b = laplacian_of_gaussian_edge_detection(b)
                edge_g = laplacian_of_gaussian_edge_detection(g)
                edge_r = laplacian_of_gaussian_edge_detection(r)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
                enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
                enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)
                # 将增强后的三个通道合并成一张彩色图像
                img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
            pass
        elif param == "Frequency Gaussian":
            if is_rgb_image(img):
                b, g, r = cv2.split(img)
                edge_b = cv2.convertScaleAbs(GaussianHighFilter(b, thresh))
                edge_g = cv2.convertScaleAbs(GaussianHighFilter(g, thresh))
                edge_r = cv2.convertScaleAbs(GaussianHighFilter(r, thresh))

                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                enhanced_b = cv2.addWeighted(b, 1, edge_b, alpha, 0)
                enhanced_g = cv2.addWeighted(g, 1, edge_g, alpha, 0)
                enhanced_r = cv2.addWeighted(r, 1, edge_r, alpha, 0)
                # 将增强后的三个通道合并成一张彩色图像
                img = cv2.merge((enhanced_b, enhanced_g, enhanced_r))
            else:
                edge = GaussianHighFilter(img, thresh)
                edge = cv2.convertScaleAbs(edge)
                # 将边缘图像应用到每个通道上
                alpha = 0.5  # 控制锐化程度的参数
                img = cv2.addWeighted(img, 1, edge, alpha, 0)

        # 更新图像显示
        show_image()

        # 关闭加载中提示窗口
        loading_window.destroy()

    # 使用线程来执行平滑处理
    Thread(target=process_sharpen).start()


def edge_detection_handler(param, thresh=100):
    global img, prev_img
    prev_img = img.copy()
    # 创建并显示加载中提示窗口
    loading_window = show_loading_window()
    def process_detect():
        global img
        nonlocal loading_window
        if param == "Roberts":
            # 处理Roberts算子逻辑
            img = roberts_edge_detection(img)
        elif param == "Sobel":
            # 处理Sobel算子逻辑
            img = sobel_edge_detection(img)
        elif param == "Prewitt":
            # 处理Prewitt算子逻辑
            img = prewitt_edge_detection(img)
        elif param == "Laplacian":
            # 处理Laplacian算子逻辑
            img = laplacian_edge_detection(img)
        elif param == "LoG":
            # 处理LoG算子逻辑
            img = laplacian_of_gaussian_edge_detection(img)
        elif param == "Canny":
            # 处理Canny边缘检测逻辑
            blur = cv2.GaussianBlur(img, (5, 5), 0)
            # 使用Canny算子进行边缘检测
            img = cv2.Canny(blur, 80, 150)
            pass
        elif param == "Frequency Gaussian":
            if is_rgb_image(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.convertScaleAbs(GaussianHighFilter(gray, thresh))
            else:
                img = cv2.convertScaleAbs(GaussianHighFilter(img, thresh))
        if is_rgb_image(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 更新图像显示
        show_image()

        # 关闭加载中提示窗口
        loading_window.destroy()

    # 使用线程来执行平滑处理
    Thread(target=process_detect).start()


def get_leaf(img):
    if not is_rgb_image(img):
        messagebox.showerror("error", "预设只接受RGB图像")
        return
    # 处理叶片分离的预设阈值分割逻辑
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
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # 进行开运算操作
    result_open = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    # 进行闭运算操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(result_open, cv2.MORPH_CLOSE, kernel)
    Mask = cv2.bitwise_not(closed)
    # 将mask三通道扩展为与img相同的通道数
    MergeMask = cv2.merge([Mask, Mask, Mask])
    # 将img与mask相乘，保留mask为255的像素，其他像素置为0
    img = cv2.bitwise_and(img, MergeMask)
    return img

def get_spot(img, threshold_value=99):
    if not is_rgb_image(img):
        messagebox.showerror("error", "预设只接受RGB图像")
        return
    b, g, r = cv2.split(img)
    # 处理叶片分离的预设阈值分割逻辑
    # 转换为HSV空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 分离通道
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(img)
    otsu_thresh = cv2.THRESH_OTSU
    _, s_th = cv2.threshold(s, 0, 255, otsu_thresh)
    _, b_th = cv2.threshold(b, 0, 255, otsu_thresh)
    # 取反
    s_th = cv2.bitwise_not(s_th)
    # 叠加
    result = cv2.bitwise_or(b_th, s_th)
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # 进行开运算操作
    result_open = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    # 进行闭运算操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(result_open, cv2.MORPH_CLOSE, kernel)
    Mask = cv2.bitwise_not(closed)
    # 将mask三通道扩展为与img相同的通道数
    mask = cv2.merge([Mask, Mask, Mask])
    leaf = img.copy()
    img = cv2.cvtColor(r, cv2.COLOR_GRAY2RGB)

    # 对图像进行掩膜处理
    masked_img = cv2.bitwise_and(img, mask)

    # 阈值分割
    ret, thresh = cv2.threshold(masked_img, threshold_value, 255, cv2.THRESH_BINARY)
    '''
    开运算消除大部分叶脉
    '''
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    # 对图像进行开运算
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    '''
    开运算消除剩余噪点
    '''
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 对图像进行开运算
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    # 对图像进行闭运算
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 对图像进行掩膜处理
    img = cv2.bitwise_and(leaf, thresh)
    return img


def threshold_handler(param, param1):
    global img, prev_img
    prev_img = img.copy()
    if param == "Leaf":
        if param1 == "Otsu":
            # 处理叶片分离的大津法阈值分割逻辑
            THRESH_OTSU(img)
            pass
        elif param1 == "Adaptive":
            # 处理叶片分离的自适应双阈值阈值分割逻辑
            img = DoubleThresholdSegmentation(img)
            pass
        elif param1 == "Preset":
            img = get_leaf(img)
            pass
    elif param == "Spot":
        if param1 == "manual":
            # 手动调整阈值
            thresh_value = 99
            # 创建新窗口
            input_window = tk.Toplevel()
            input_window.title("输入窗口")
            flag = False
            def close_window():
                nonlocal flag
                input_window.destroy()
                flag = True
                # 直接退出 morphology_operation 函数
                return

            # 注册关闭事件
            input_window.protocol("WM_DELETE_WINDOW", close_window)

            def process_inputs(input1):
                nonlocal thresh_value
                thresh_value = int(input1)
                input_window.destroy()
                return input1


            # 创建输入框和确认按钮
            input_label1 = tk.Label(input_window, text="阈值:")
            input_label1.pack()

            input_entry1 = tk.Entry(input_window)
            input_entry1.pack()

            confirm_button = tk.Button(input_window, text="确认",
                                       command=lambda: process_inputs(input_entry1.get()))
            confirm_button.pack()
            # 等待窗口关闭
            input_window.wait_window()
            if flag:
                return
            img = get_spot(img, threshold_value=thresh_value)
            pass
        elif param1 == "Preset":
            # 处理病斑分离的预设阈值分割逻辑
            img = get_spot(img)

            pass
    show_image()


# 形态学操作处理函数
def morphology_operation(operation):
    global img, prev_img

    kernel_shape = "MORPH_ELLIPSE"
    kernel_x = 3
    kernel_y = 3
    # 创建新窗口
    input_window = tk.Toplevel()
    input_window.title("输入窗口")
    def process_inputs(shape, input1, input2):
        global kernel_shape, kernel_x, kernel_y
        kernel_shape = shape
        kernel_x = int(input1)
        kernel_y = int(input2)
        input_window.destroy()
        return shape, input1, input2
    flag = False
    def close_window():
        nonlocal  flag
        input_window.destroy()
        # 直接退出 morphology_operation 函数
        flag = True
        return
    # 注册关闭事件
    input_window.protocol("WM_DELETE_WINDOW", close_window)
    # 创建下拉选择框
    combobox_label = tk.Label(input_window, text="下拉选择框:")
    combobox_label.pack()

    values = ["MORPH_ELLIPSE", "MORPH_RECT"]  # 可供选择的选项列表

    combobox = Combobox(input_window, values=values)
    combobox.set(values[0])
    combobox.pack()

    # 创建输入框和确认按钮
    input_label1 = tk.Label(input_window, text="结构元素x:")
    input_label1.pack()

    input_entry1 = tk.Entry(input_window)
    input_entry1.pack()

    input_label2 = tk.Label(input_window, text="结构元素y:")
    input_label2.pack()

    input_entry2 = tk.Entry(input_window)
    input_entry2.pack()

    confirm_button = tk.Button(input_window, text="确认",
                               command=lambda: process_inputs(combobox.get(), input_entry1.get(), input_entry2.get()))
    confirm_button.pack()
    # 等待窗口关闭
    input_window.wait_window()
    if flag:
        return
    prev_img = img.copy()
    # 创建子窗口，添加下拉框、输入框、确认按钮等组件，并实现相应的逻辑
    if kernel_shape == "MORPH_ELLIPSE":
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_x, kernel_y))
    else:
        # 定义结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_x, kernel_y))
    if not is_binary_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        messagebox.showwarning("Warning", "RGB图像强制转换为二值图像")
        pass
    if operation == "Erosion":
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        pass
    elif operation == "Dilation":
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
        pass
    elif operation == "Opening":
        # 对图像进行开运算
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        pass
    elif operation == "Closing":
        # 对图像进行闭运算
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        pass
    show_image()


# 掩模处理函数
def mask_operation(mask_type):
    global img, prev_img
    prev_img = img.copy()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        tmp = cv2.imread(file_path)
    else:
        return
    if not is_binary_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        messagebox.showwarning("Warning", "RGB图像强制转换为二值图像")
        pass
    else:
        messagebox.showerror("error","不合理的路径")
    if mask_type == "Bright":
        b, g, r = cv2.split(tmp)
        img_b = cv2.bitwise_and(b, img)
        img_g = cv2.bitwise_and(g, img)
        img_r = cv2.bitwise_and(r, img)
        img = cv2.merge([img_b, img_g, img_r])
        pass
    elif mask_type =="Dark":
        img = cv2.bitwise_not(img)
        b, g, r = cv2.split(tmp)
        img_b = cv2.bitwise_and(b, img)
        img_g = cv2.bitwise_and(g, img)
        img_r = cv2.bitwise_and(r, img)
        img = cv2.merge([img_b, img_g, img_r])
        pass
    show_image()
    # 检测当前img是否是二值图像，如果不是，转换为二值图像
    # 根据mask_type进行掩模操作，保存结果到Mask变量中
    # 根据原始图路径重新读取图片，保存到img变量中
    # 对img和Mask使用bitwise_and取出亮区或暗区
    # 在下方窗口中显示变换后的图片img
    pass

# Kmeans处理函数
def kmeans_operation(operation):
    # 检测当前img是否是RGB图片，如果不是，转换为RGB三通道图片
    # 执行相应的Kmeans聚簇算法函数（预设函数标识）
    # 显示加载动画，直到函数执行完成
    # 将运算结果保存到img变量中
    # 在下方窗口中显示变换后的图片img
    global img, prev_img
    prev_img = img.copy()
    if not is_rgb_image(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # 创建并显示加载中提示窗口
    loading_window = show_loading_window()

    def kmeans():
        global img
        nonlocal loading_window
        if operation == "Pseudocolor":
            img = get_leaf(img)
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
            img = new_img.reshape(height, width, 3)
            pass
        elif operation == "Lesion":
            img = get_leaf(img)
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
            # 提取黑色像素
            black_pixels = np.all(new_img == [0, 0, 0], axis=-1)
            copy_img = np.zeros_like(new_img)
            # 将黑色像素设置为白色，其他像素设置为黑色
            copy_img[black_pixels] = [255, 255, 255]
            copy_img[~black_pixels] = [0, 0, 0]
            img = copy_img
            pass

        # 更新图像显示
        show_image()

        # 关闭加载中提示窗口
        loading_window.destroy()

    # 使用线程来执行平滑处理
    Thread(target=kmeans).start()
    pass



# 创建菜单栏
menu_bar = tk.Menu(root)

# 文件菜单栏
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="打开图片", command=open_image)
save_menu = tk.Menu(file_menu, tearoff=0)
file_menu.add_cascade(label="保存图片", command=save_image)
reset_menu = tk.Menu(file_menu, tearoff=0)
file_menu.add_cascade(label="重置图片", command=reset_image)
menu_bar.add_cascade(label="文件", menu=file_menu)

# 灰度菜单栏
gray_menu = tk.Menu(menu_bar, tearoff=0)
gray_menu.add_command(label="灰度图", command=convert_to_gray)
gray_menu.add_command(label="灰度直方图", command=show_histogram)
menu_bar.add_cascade(label="灰度", menu=gray_menu)

# 色彩空间菜单栏
color_menu = tk.Menu(menu_bar, tearoff=0)
channel_menu = tk.Menu(color_menu, tearoff=0)
channel_menu.add_command(label="R通道图", command=lambda: channel_handler('R'))
channel_menu.add_command(label="R通道直方图", command=lambda: channel_handler('R', True))
color_menu.add_cascade(label="R通道", menu=channel_menu)

channel_menu = tk.Menu(color_menu, tearoff=0)
channel_menu.add_command(label="G通道图", command=lambda: channel_handler('G'))
channel_menu.add_command(label="G通道直方图", command=lambda: channel_handler('G', True))
color_menu.add_cascade(label="G通道", menu=channel_menu)

channel_menu = tk.Menu(color_menu, tearoff=0)
channel_menu.add_command(label="B通道图", command=lambda: channel_handler('B'))
channel_menu.add_command(label="B通道直方图", command=lambda: channel_handler('B', True))
color_menu.add_cascade(label="B通道", menu=channel_menu)

channel_menu = tk.Menu(color_menu, tearoff=0)
channel_menu.add_command(label="H通道图", command=lambda: channel_handler('H'))
channel_menu.add_command(label="H通道直方图", command=lambda: channel_handler('H', True))
color_menu.add_cascade(label="H通道", menu=channel_menu)

channel_menu = tk.Menu(color_menu, tearoff=0)
channel_menu.add_command(label="S通道图", command=lambda: channel_handler('S'))
channel_menu.add_command(label="S通道直方图", command=lambda: channel_handler('S', True))
color_menu.add_cascade(label="S通道", menu=channel_menu)

channel_menu = tk.Menu(color_menu, tearoff=0)
channel_menu.add_command(label="V通道图", command=lambda: channel_handler('V'))
channel_menu.add_command(label="V通道直方图", command=lambda: channel_handler('V', True))
color_menu.add_cascade(label="V通道", menu=channel_menu)

menu_bar.add_cascade(label="色彩空间", menu=color_menu)

# 平滑菜单栏
smooth_menu = tk.Menu(menu_bar, tearoff=0)
smooth_menu.add_command(label="空间域高斯滤波", command=lambda: smooth_handler('Spatial Gaussian'))
smooth_menu.add_command(label="均值滤波", command=lambda: smooth_handler('Mean'))
smooth_menu.add_command(label="中值滤波", command=lambda: smooth_handler('Median'))
smooth_menu.add_command(label="频率域高斯低通滤波", command=lambda: smooth_handler('Frequency Gaussian'))
menu_bar.add_cascade(label="平滑", menu=smooth_menu)

# 锐化菜单栏
sharpen_menu = tk.Menu(menu_bar, tearoff=0)


sharpen_menu.add_command(label="Roberts", command=lambda: sharpen_handler('Roberts'))
sharpen_menu.add_command(label="Sobel", command=lambda: sharpen_handler('Sobel'))
sharpen_menu.add_command(label="Prewitt", command=lambda: sharpen_handler('Prewitt'))
sharpen_menu.add_command(label="Laplacian", command=lambda: sharpen_handler('Laplacian'))
sharpen_menu.add_command(label="LoG", command=lambda: sharpen_handler('LoG'))
sharpen_menu.add_command(label="频率域高斯高通锐化", command=lambda: sharpen_handler('Frequency Gaussian'))
menu_bar.add_cascade(label="锐化", menu=sharpen_menu)

# 边缘检测菜单栏
edge_detection_menu = tk.Menu(menu_bar, tearoff=0)
edge_detection_menu.add_command(label="Roberts", command=lambda: edge_detection_handler('Roberts'))
edge_detection_menu.add_command(label="Sobel", command=lambda: edge_detection_handler('Sobel'))
edge_detection_menu.add_command(label="Prewitt", command=lambda: edge_detection_handler('Prewitt'))
edge_detection_menu.add_command(label="Laplacian", command=lambda: edge_detection_handler('Laplacian'))
edge_detection_menu.add_command(label="LoG", command=lambda: edge_detection_handler('LoG'))
edge_detection_menu.add_command(label="Canny", command=lambda: edge_detection_handler('Canny'))
edge_detection_menu.add_command(label="频率域高斯高通滤波", command=lambda: edge_detection_handler('Frequency Gaussian'))
menu_bar.add_cascade(label="边缘检测", menu=edge_detection_menu)

# 阈值分割菜单栏
threshold_menu = tk.Menu(menu_bar, tearoff=0)
leaf_menu = tk.Menu(threshold_menu, tearoff=0)
leaf_menu.add_command(label="大津法", command=lambda: threshold_handler('Leaf', 'Otsu'))
leaf_menu.add_command(label="自适应双阈值", command=lambda: threshold_handler('Leaf', 'Adaptive'))
leaf_menu.add_command(label="预设", command=lambda: threshold_handler('Leaf', 'Preset'))
threshold_menu.add_cascade(label="叶片分离", menu=leaf_menu)
spot_menu = tk.Menu(threshold_menu, tearoff=0)
spot_menu.add_command(label="手动调整阈值", command=lambda: threshold_handler('Spot', 'manual'))
spot_menu.add_command(label="预设", command=lambda: threshold_handler('Spot', 'Preset'))
threshold_menu.add_cascade(label="病斑分离", menu=spot_menu)
menu_bar.add_cascade(label="阈值分割", menu=threshold_menu)


# 形态学操作菜单栏
morphology_menu = tk.Menu(menu_bar, tearoff=0)
morphology_menu.add_command(label="腐蚀", command=lambda: morphology_operation('Erosion'))
morphology_menu.add_command(label="膨胀", command=lambda: morphology_operation('Dilation'))
morphology_menu.add_command(label="开运算", command=lambda: morphology_operation('Opening'))
morphology_menu.add_command(label="闭运算", command=lambda: morphology_operation('Closing'))
# morphology_menu.add_command(label="高帽运算", command=lambda: morphology_operation('TopHat'))
# morphology_menu.add_command(label="低帽运算", command=lambda: morphology_operation('BlackHat'))
menu_bar.add_cascade(label="形态学操作", menu=morphology_menu)


# 掩模菜单栏
mask_menu = tk.Menu(menu_bar, tearoff=0)
mask_menu.add_command(label="保留亮区", command=lambda: mask_operation('Bright'))
mask_menu.add_command(label="保留暗区", command=lambda: mask_operation('Dark'))
menu_bar.add_cascade(label="掩模", menu=mask_menu)


# Kmeans菜单栏
kmeans_menu = tk.Menu(menu_bar, tearoff=0)
kmeans_menu.add_command(label="伪彩色聚簇", command=lambda: kmeans_operation('Pseudocolor'))
kmeans_menu.add_command(label="提取病斑", command=lambda: kmeans_operation('Lesion'))
menu_bar.add_cascade(label="Kmeans", menu=kmeans_menu)




# 绑定菜单栏
root.config(menu=menu_bar)

# 图像显示区域
image_label = tk.Label(root)
image_label.pack()

stop_flag = False

def keyboard_listener():
    # 监听键盘事件，当按下 Ctrl+Z 时调用 ctrl_z_handler 函数
    add_hotkey("ctrl+z", ctrl_z_handler)
    # 启动监听
    global stop_flag
    if stop_flag:
        return


# 创建并启动键盘监听线程
keyboard_thread = Thread(target=keyboard_listener)
keyboard_thread.start()

def ctrl_z_handler():
    # 处理 Ctrl+Z 快捷键的操作
    pass


def close_window():
    # 停止键盘监听
    unhook_all()

    # 设置 stop_flag 为 True，通知线程停止等待
    global stop_flag
    stop_flag = True

    # 等待键盘监听线程执行完毕
    keyboard_thread.join()

    # 销毁窗口
    root.destroy()


# 运行主循环
root.mainloop()
