from show import img_show
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


img = cv2.imread(r'E:\pyitem\opencv_img\photo\canon.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img_gray)
# img_show(img)

# # 二值化
# x = np.random.randint(1, 10, (3, 3))
# print(x)
# x[x > 5] = 255
# x[x <= 5] = 0
# print(x)


# x = np.arange(2, 20)
# y = 2*x + np.random.randint(5, 10, 18)
# plt.plot(x, y, 'm+')
# plt.show()
# 图像的通道分离
# b, g, r = cv2.split(img)
# print(b.shape)  # (266, 400)

# # 绘制网格图形
# x1 = np.linspace(0, 1, 100)
# y1 = np.power(x1, 0.5)
# y2 = x1
# y3 = np.power(x1, 2)
# plt.plot(x1, y1, label='0.5')
# plt.plot(x1, y2, label='1')
# plt.plot(x1, y3, label='2')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid()
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.show()


# # 绘制直方图
# a = np.random.randint(0, 256, 1000)
# bins = np.arange(-0.5, 256, 1)
# # print(bins)
# plt.hist(a)
# plt.show()
# plt.hist(a, bins, rwidth=0.9)
# plt.show()


# 图像的线性变换
# # print(img.dtype)  # uint8
# k = 2
# b = 20
# img_line = b + k * img.astype(np.int32)
# # print(img_line.min())
# # img_show(img_line)
# img_line1 = np.clip(img_line, 0, 255)
# img_line1 = img_line1.astype(np.uint8)
# # print(img_line1, img_line1.dtype, img_line1.shape)
# img_line2 = cv2.convertScaleAbs(img, alpha=2, beta=20)
# # print(img_line2)
# img_show(np.hstack([img_line2, img_line1]))

# 图像的非线性变换
# a = 10
# b = 0.1
# img_uline = a + np.log(img.astype(np.float64)+1) / b
# img_uline = img_uline.astype(np.uint8)
# img_show(img_uline)

# 图像的r变换
# img0 = img / 255
# img05 = np.power(img0, 0.5) * 255
# img05 = img05.astype(np.uint8)
# img15 = np.power(img0, 1.5) * 255
# img15 = img15.astype(np.uint8)
# img_show(np.hstack([img05, img, img15]))

# img_resize = cv2.resize(img, (266, 400))
# print(img_resize.shape)  # (400, 266, 3)
# img_show(img_resize)

# # 图像的错切变换
# M = np.array([[1, np.tan(0.5), 0], [0, 1, 0]], dtype=np.float32)
# img_c = cv2.warpAffine(img, M, (400, 266))
# img_show(np.hstack([img, img_c]))

# 图像的水平镜像
# M = np.array([[-1, 0, 400], [0, 1, 0]], dtype=np.float32)
# img_c = cv2.warpAffine(img, M, (400, 266))
# img_show(np.hstack([img_c, img]))

# # opencv自带镜像函数
# img_h = cv2.flip(img, 0)  # 垂直镜像
# img_l = cv2.flip(img, 1)  # 水平镜像
# img_hl = cv2.flip(img, -1) # 水平垂直镜像
# img_show(img_h)
# img_show(img_l)
# img_show(img)

# # 图像按角度旋转
# img_chengji = cv2.imread(r'E:\pyitem\opencv_img\photo\chengji.jpg')
# # print(img_chengji.shape)  # (1362, 960, 3)
# a = 5*np.pi/180
# M = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0]], dtype=np.float32)
# img_x = cv2.warpAffine(img_chengji, M, (960, 1362))
# img_show(np.hstack([img_chengji, img_x]))

# # 按照图像中心为轴旋转
# img_chengji = cv2.imread(r'E:\pyitem\opencv_img\photo\chengji.jpg')
# h, w, c = img_chengji.shape
# M = cv2.getRotationMatrix2D((w//2, h//2), 3, 1)
# img_x = cv2.warpAffine(img_chengji, M, (1000, 1362))
# cv2.imwrite(r'E:\pyitem\opencv_img\photo\chengji_n.jpg', img_x)
# # img_show(np.hstack([img_chengji, img_x]))

# # opencv 自带函数库旋转 90,180度
# img_c90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# img_show(img_c90)

# # 图像的透视变换
# img_chengji = cv2.imread(r'E:\pyitem\opencv_img\photo\chengji.jpg')
# # img_show(img_chengji)
# src = np.array([[50, 0],
#                 [960, 0],
#                 [0, 1350],
#                 [960, 1362]], dtype=np.float32)
# dst = np.array([[0, 0],
#                 [950, 0],
#                 [0, 1362],
#                 [960, 1362]], dtype=np.float32)
# M = cv2.getPerspectiveTransform(src, dst)
# img_chengji_p = cv2.warpPerspective(img_chengji, M, (960, 1362))
# cv2.imwrite(r'E:\pyitem\opencv_img\photo\chengji_p.jpg', img_chengji_p)
# # img_show(img_chengji_p)


# # 双边滤波算法
# def filter(img, size, d, r):
#     pad_num = size // 2  # 边缘扩充尺寸
#     img_pad = np.pad(img, pad_num, mode='constant', constant_values=0)  # 对图像边缘扩充
#     # 产生空间核算子
#     kernal_c = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             kernal_c[i, j] = np.exp(-0.5*((i-pad_num)**2+(j-pad_num)**2)/(d**2))
#     kernal_c/=kernal_c.sum()
#     # 产生像素核算子
#     kernal_s = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             roi = img[i:i + size, j:j + size]
#             kernal_s[i, j] = np.exp(-0.5*((roi[i,j]-roi[pad_num, pad_num])**2)/(r**2))
#     kernal_s/=kernal_s.sum()
#     h, w, c = img_pad.shape
#     for i in range(h-pad_num):
#         for j in range(w-pad_num):
#             roi = img[i:i + size, j:j + size]
#             k = kernal_c * kernal_s
#             k/=k.sum()
#             img[i, j] = (roi * k).sum
#     return img

# img_double_filter = filter(img, 11, 5, 3)
# img_show(img_double_filter)

# # 空间核算子
# def get_c(size, d):
#     pad_num = (size-1)/2  # 边缘扩充尺寸
#     # 产生空间核算子
#     kernal_c = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             kernal_c[i, j] = np.exp(-0.5*((i-pad_num)**2+(j-pad_num)**2)/(d**2))
#     kernal_c = kernal_c / kernal_c.sum()
#     return kernal_c
# c = get_c(11, 3)
# # img_show(c)


# # 产生像素核算子(自己写的)
# def get_s(img, size, r):
#     img = np.float64(img)
#     pad_num = size // 2  # 边缘扩充尺寸
#     kernal_s = np.zeros((size, size))
#     for i in range(size):
#         for j in range(size):
#             kernal_s[i, j] = np.exp(-0.5 * ((img[i, j] - img[pad_num, pad_num]) ** 2) / (r ** 2))
#     kernal_s/=kernal_s.sum()
#     return kernal_s
# s = get_s(c, 11, 10)
# img_show(s)


# # 产生像素核算子
# def get_s(f, sigmar, n):
#     s = np.zeros((n, n))
#     f = np.float64(f)
#     for i in range(n):
#         for j in range(n):
#             s[i, j] = np.exp(-0.5 * ((f[i, j] - f[n//2, n//2]) ** 2) / (sigmar ** 2))
#     s/=s.sum()
#     return s
# s = get_s(c, 11, 10)
# img_show(s)


# # prewitt算法
# #水平方向算子
# kernal_px = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
# #垂直方向算子
# kernal_py = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
# img_X = cv2.filter2D(img, cv2.CV_64F, kernal_px)
# img_X = np.uint8(np.abs(img_X).clip(0, 255))
# img_Y = cv2.filter2D(img, cv2.CV_64F, kernal_py)
# img_Y = np.uint8(np.abs(img_Y).clip(0, 255))
# img_XY = (np.abs(img_X) + np.abs(img_Y)).clip(0, 255)
# img = (img + img_Y).clip(0, 255)
# img_show(np.hstack([img_XY]))


# 直方图分割阈值
img_td = cv2.imread(r'E:\pyitem\opencv_img\photo\tiedao.jpg')
gray = cv2.cvtColor(img_td, cv2.COLOR_BGR2GRAY)
# # 自己写的分割算法
# plt.hist(img_td.ravel(), 256, (0, 256))
# plt.hist(gray.ravel(), 256, (0, 256))
# plt.show()
# h, w, c = img_td.shape
# for i in range(h):
#     for j in range(w):
#         if gray[i, j] >= 185:
#             gray[i, j] = 255
#         else:
#             gray[i, j] = 0
# img_show(gray)
# # opencv自带的分割算法
# _, img_td_bin = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
# img_show(img_td_bin)


# # 迭代法阈值分割 (自己写的)
# gray_mean = np.uint8(gray.mean())
# h, w = gray.shape
# while True:
#     # qian = np.zeros_like(gray, dtype=np.float32)  # 存储前景信息 该存储方式不推荐
#     qian = 0  # 推荐使用该存储方式
#     qian_num = 0  # 用于前景信息计数
#     # bei = np.zeros_like(gray, dtype=np.float32)  # 存储背景信息
#     bei = 0
#     bei_num = 0  # 用于背景信息计数
#     for i in range(h):
#         for j in range(w):
#             if gray[i, j] >= gray_mean:
#                 qian = qian + gray[i, j]
#                 # qian[i, j] = gray[i, j] # 不推荐
#                 qian_num += 1
#             else:
#                 bei = bei + gray[i, j]
#                 # bei[i, j] = gray[i, j]  # 不推荐
#                 bei_num += 1
#     #计算前景背景的灰度平均值
#     qian_mean = (qian / qian_num)
#     bei_mean = (bei / bei_num)
#     gray_mean_new = np.uint8((qian_mean+bei_mean)/2)
#     # 该方式不推荐
#     # qian_mean = (qian.sum() / qian_num)
#     # bei_mean = (bei.sum() / bei_num)
#     # gray_mean_new = np.uint8((qian_mean+bei_mean)/2)
#     if gray_mean_new == gray_mean:
#         break
#     else:
#         gray_mean = gray_mean_new
#         continue
# print(gray_mean_new)


# # 迭代法阈值分割(简洁写法)
# start = time.perf_counter()
# img_td_mean = img_td.mean()
# while True:
#     bei_mean = img_td[img_td < img_td_mean].mean()
#     qian_mean = img_td[img_td >= img_td_mean].mean()
#     img_td_mean_new = (bei_mean+qian_mean)/2
#     if img_td_mean_new == img_td_mean:
#         break
#     else:
#         img_td_mean = img_td_mean_new
#         continue
# end = time.perf_counter()
# print(f'run time is {end-start}')
# print(np.uint8(img_td_mean))


# 大津法阈值 opencv自带函数
# start = time.perf_counter()
# thresh, img_bin = cv2.threshold(gray, -1, 255, cv2.THRESH_OTSU)
# end = time.perf_counter()
# print(f'running time is {end-start}')
# print(thresh)
# img_show(img_bin)

# # 手写大津法
# start = time.perf_counter()
# T = 0  # 存放阈值
# sigma = -1  # 存放方差
# h, w = gray.shape
# for t in range(1, 250):
#     qian = 0  # 存储前景像素值
#     qian_num = 0  # 前景个数
#     bei = 0  # 存储背景像素值
#     bei_num = 0  # 背景个数
#     for i in range(h):
#         for j in range(w):
#             if gray[i, j] > t:
#                 qian_num+=1
#                 qian = qian + gray[i, j]
#             else:
#                 bei_num+=1
#                 bei = bei + gray[i, j]
#     m0 = qian / qian_num  # 前景平均灰度值
#     m1 = bei / bei_num   # 背景平均灰度值
#     p0 = qian_num / (qian_num + bei_num)  # 前景所占比例
#     p1 = bei_num / (qian_num + bei_num)
#     m_mean = p0 * m0 + p1 * m1   # 整图平均灰度值
#     sigma_new = p0*p1*(m0-m1)*(m0-m1)  # 类间方差
#     if sigma_new >= sigma:
#         sigma = sigma_new
#         T = t
#     else:
#         continue
# end = time.perf_counter()
# print(f'running time is {end-start}')
# print(sigma, T)


# # 手写大津法 （简洁写法）
# start = time.perf_counter()
# T = 0
# sigma = -1
# for t in range(0, 256):
#     qian = gray[gray >= t]
#     bei = gray[gray < t]
#     m0 = qian.mean()
#     m1 = bei.mean()
#     p0 = qian.size / gray.size
#     p1 = bei.size / gray.size
#     sigma_new = p0*p1*(m0-m1)**2
#     if sigma_new >= sigma:
#         sigma = sigma_new
#         T = t
#     else:
#         continue
# end = time.perf_counter()
# print(f'running time is {end-start}')
# print(sigma, T)


# # 手写局部二值化
# # img_blur = cv2.GaussianBlur(img_td, (5, 5), 2)
# # img_show(img_blur)
# gray_img = cv2.cvtColor(img_td, cv2.COLOR_BGR2GRAY)
# h, w = gray_img.shape
# img_pad = np.pad(gray_img, 3, mode='constant', constant_values=0)
# for i in range(3, h-3):
#     for j in range(3, w-3):
#         roi = img_pad[i-3:i+3+1, j-3:j+3+1]  # 提取roi区域
#         roi_mean = roi.mean()
#         if img_pad[i, j] > roi_mean:
#             img_pad[i, j] = 255
#         else:
#             img_pad[i, j] = 0
# # 将处理后的图像裁剪为原图大小
# img_bin = img_pad[3:h-3, 3:w-3]
# img_show(img_bin)


# opencv自带自适应阈值函数
img_blur = cv2.GaussianBlur(img_td, (5, 5), 2)
gray_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
img_bins = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
img_show(img_bins)
