# # import tkinter as tk
# # import cv2
# #
# # # 创建窗口
# # window = tk.Tk()
# # window.title('window')
# # window.geometry('600x600')   # 窗口大小
# # var = tk.StringVar()
# # l = tk.Label(window, textvariable=var, bg='green', width=12, height=2)
# # l.pack()
# #
# # on_hit = True
# # # def hit_me():
# # #     global on_hit
# # #     if on_hit:
# # #         on_hit = False
# # #         var.set('我叫魏振萌')
# # #     else:
# # #         on_hit = True
# # #         var.set('')
# #
# # def hit_me():
# #     global on_hit
# #     if on_hit:
# #         on_hit = False
# #         img = cv2.imread(r'E:\pyitem\opencv_img\photo\canon.jpg')
# #         cv2.imshow('img', img)
# #         cv2.waitKey(0)
# #         cv2.destroyAllWindows()
# #         # var.set(img)
# #     else:
# #         on_hit = True
# #         var.set('')
# #
# # # 设置按钮
# # b = tk.Button(window, text='hit me', width=15, height=2, command=hit_me)
# # b.pack()
# # window.mainloop()
#
# # import cv2
# #
# # img = cv2.imread("./photo/pyramid.jpg")
# # # img = cv2.imread("./photo/head.jpg")
# # #遍历图像某个点的像素值
# # # for i in range(img.shape[0]):
# # #     for j in range(img.shape[1]):
# # #         x = img[i][j]
# # #         x1 = img[i, j]
# # #         print(x)
# # #         print(x1)
# # #         break
# # #     break
# #
# # #比特分层处理
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # # print(gray.shape)
# # # print(img.shape)
# # #取出各个比特层
# # # two_bit = gray & 0x02
# # five_bit = gray & 0x10
# # six_bit = gray & 0x20
# # seven_bit = gray & 0x40
# # eight_bit = gray & 0x80
# #
# # # cv2.imshow('two_bit', two_bit)
# # # cv2.imshow('five_bit', five_bit)
# # # cv2.imshow('six_bit', six_bit)
# # # cv2.imshow('seven_bit', seven_bit)
# # cv2.imshow('eight_bit', eight_bit)
# #
# #
# # # print(gray[13, 10])
# # # print(two_bit[13, 10])
# # # print(five_bit[13, 10])
# # # print(six_bit[13, 10])
# # # print(seven_bit[13, 10])
# # print(eight_bit[181, 246])
# #
# #
# # # print(eight_bit)
# #
# # # for i in range(seven_bit.shape[0]):
# # #     for j in range(seven_bit.shape[1]):
# # #         if (eight_bit[i, j]==128):
# # #             print(i, j)
# #
# # # gray_678 = five_bit + six_bit + seven_bit + eight_bit
# # # cv2.imshow('original', img)
# # # cv2.imshow('gray', gray)
# # # cv2.imshow('gray_678', gray_678)
# # # print(gray_678[13, 10])
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
#
# import numpy as np
# import cv2
#
# # 十进制转换二进制
# def num_bit(num, b):
#     i = 0
#     # for i in range(8):
#     #     b[i] = 0
#
#     # while(num):
#     #     b.append(np.uint8(num % 2))
#     #     num = np.floor(num / 2)
#     #     i = i + 1
#
#     while(num):
#         b[i] = np.uint8(num % 2)
#         num = np.floor(num / 2)
#         i+=1
#     return b
#
# def img_bit(img, num):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     b = np.zeros(8)
#
#     gray_b = np.zeros((gray.shape[0], gray.shape[1]))
#     for i in range(gray.shape[0]):
#         for j in range(gray.shape[1]):
#             b = num_bit(gray[i, j], b)  # 将每点的像素值转换为二进制形式
#             gray_b[i, j] = b[num-1]
#             b = np.zeros(8)
#
#     return gray_b
#     # return b.shape
#
#
# def select(arr, len):
#
#
#     for i in range(len):
#         max = i
#         for j in range(i+1, len):
#             if (arr[j]>arr[max]):
#                 max = j
#         if(max != i):
#             temp = arr[max]
#             arr[max] = arr[i]
#             arr[i] = temp
#
#     return arr
#     # print(arr)
#
#
# if __name__ == '__main__':
#     img = cv2.imread("./photo/pyramid.jpg")
#     # b = np.zeros(8)
#     # print(num_bit(2, b))
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # print(gray)
#     # five = img_bit(img, 5)*16
#     # six = img_bit(img, 6)*32
#     # seven = img_bit(img, 7)*64
#     # eight = img_bit(img, 8)*128
#     # gray_b = (five + six + seven + eight)/256
#
#     # print(gray_b)
#     # cv2.imshow('gray_b', gray_b)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     # arr = [5, 8, 9, 6, 1]
#     # len = 5
#     # print(arr)
#     # # select(arr, len)
#     # arr = select(arr, len)
#     # print(arr)
#
#     img = np.uint8(img)
#     sum = cv2.integral(img)
#     cv2.imshow("integral", sum)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#
#


import cv2
import numpy as np
from PIL import Image

# image = Image.open("./photo/zd.png")
# img = cv2.imread('./photo/zd.png', 0)
# img1 = cv2.imread('./photo/zd.png', cv2.IMREAD_UNCHANGED)
# print(img.shape)
# print(img1.shape)


# image = np.array(image, np.uint8)
# rows,cols,dims=image.shape
#
# sum = np.zeros((rows,cols),np.int32)
# imageIntegral = cv2.integral(image, sum,-1)
# cv2.imshow("Integral Image",imageIntegral)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# a = np.array([[1, 2, 3], [9, 7, 8]])
# # max = np.max(a, axis=1)
# # mean = np.mean(a, axis=0)
# # print(mean)
#
# train = a[:, -1]
# print(train)

# ##############sobel算子##################
# def sobel_x(img):
#     g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     img_padding = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
#     for i in range(1, img_padding.shape[0]-1):
#         for j in range(1, img_padding.shape[1]-1):
#             roi = img_padding[i-1:i+1+1, j-1:j+1+1]
#             img_padding[i][j] =(roi * g_x).sum()
#             # print(img_padding)
#             if(img_padding[i][j]>170):
#                 img_padding[i][j] = 255
#             else:
#                 img_padding[i][j] = 0
#     return img_padding
#
#
#
# def sobel_y(img):
#     g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#     img_padding = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
#     for i in range(1, img_padding.shape[0]-1):
#         for j in range(1, img_padding.shape[1]-1):
#             roi = img_padding[i-1:i+1+1, j-1:j+1+1]
#             img_padding[i][j] =(roi * g_y).sum()
#             # print(img_padding)
#             if(img_padding[i][j]>170):
#                 img_padding[i][j] = 255
#             else:
#                 img_padding[i][j] = 0
#     return img_padding
#
# if __name__ == '__main__':
#     # g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     # x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     # print((g_x*x).sum())
#     img = cv2.imread('./photo/canon.jpg')
#     print(type(img))
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # # sobel(gray)
#     # img_x = sobel_x(gray)
#     # img_y = sobel_y(gray)
#     # cv2.imshow('imgx', img_x)
#     # cv2.imshow('imgy', img_y)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

import cv2
import numpy as np

# img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\ROI.jpg')
# print(img[1][1])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
# print(binary)

# img_aug = np.zeros((img.shape[0], img.shape[1], 3))
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         if(img[i][j].all()<10):
#             img_aug[i][j] = img[i][j]*1.5
#         else:
#             img_aug[i][j] = img[i][j]

# img_aug = np.power(img, 0.1)

# print(img_aug)


# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15, param1=50, param2=10, minRadius=6, maxRadius=15)  # param2 值越小，检测到的元越多
#                                                       # 返回值为圆心坐标及半径
# circles = np.uint16(np.around(circles))
# # print(circles)
# print('钢管数量：', len(circles[0]))
# for i in circles[0]:
#     # cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
#     cv2.circle(img, (i[0], i[1]), 1, (0, 255, 255), 2)
# cv2.imshow('circle_detection', img)
# # cv2.imshow('ori', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# draw_img = img.copy()
# cv2.drawContours(draw_img, contours, -1, (255, 0, 0), 2)


# #透视校正
# ROTATED_SIZE_W = 950  # 透视变换后的表盘图像大小
# ROTATED_SIZE_H = 420  # 透视变换后的表盘图像大小
# # 原图中书本的四个角点(左上、右上、右下、左下),与变换后矩阵位置
# # pts1 = np.float32([[63, 72], [163, 32], [268, 144], [150, 215]])
# pts1 = np.float32([[20, 195], [963,223], [959,560], [14,615]])
# # 变换后矩阵位置
# pts2 = np.float32([[0, 0], [ROTATED_SIZE_W, 0], [ROTATED_SIZE_W, ROTATED_SIZE_H], [0, ROTATED_SIZE_H], ])
# # 生成透视变换矩阵；进行透视变换
# M = cv2.getPerspectiveTransform(pts1, pts2)
# result_img = cv2.warpPerspective(img, M, (ROTATED_SIZE_W, ROTATED_SIZE_H))

# #检测圆
# gray1 = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
#
# circles = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1, 15, param1=50, param2=10, minRadius=8, maxRadius=20)  # param2 值越小，检测到的元越多
#                                                       # 返回值为圆心坐标及半径
# circles = np.uint16(np.around(circles))
# # print(circles)
# print('圆的个数为：', len(circles[0]))
# for i in circles[0]:
#     cv2.circle(result_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
#     cv2.circle(result_img, (i[0], i[1]), 1, (0, 255, 255), 2)
# cv2.imshow('circle_detection', result_img)


# #添加进度条
# cv2.namedWindow('Canny')
# #回调函数
# def nothing(x):
#     pass
# cv2.createTrackbar('threshold1', 'Canny', 50, 200, nothing)
# cv2.createTrackbar('threshold2', 'Canny', 10, 50, nothing)
# while(1):
#     threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
#     threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
#     img_edges = cv2.Canny(gray, threshold1, threshold2)
#     cv2.imshow('Canny', img_edges)
#     if cv2.waitKey(1)==ord('q'):
#         break
#
# cv2.destroyAllWindows()


# #查看图片前景背景位置
# img = cv2.imread(r'E:\pyitem\opencv_img\photo\gangguan2.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# qian = img[225:245, 20:40]
# bei = img[0:20, 680:700]
# cv2.imshow('qian', qian)
# cv2.imshow('bei', bei)

# print(img.shape[0]//32)

# ##添加进度条
# cv2.namedWindow('Canny')
# def nothing(x):
#     pass
#
# cv2.createTrackbar('threshold1', 'Canny', 10, 100, nothing)
# cv2.createTrackbar('threshold2', 'Canny', 50, 300, nothing)
# while(1):
#     threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
#     threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
#     img_edges = cv2.Canny(gray, threshold1, threshold2)
#     cv2.imshow('Canny', img_edges)
#     if cv2.waitKey(1)==ord('q'):
#         break
#
# cv2.destroyAllWindows()


# cv2.imshow('contours', result_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#-------------------------------------------------------------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def load_data():
    pwd = os.getcwd()

    pos_dir = os.path.join(pwd, r'E:\pyitem\SVM\SVM_detectCircle\Lenet\train\Positive1')  #E:\pyitem\SVM\SVM_detectCircle\Lenet\train\
    if os.path.exists(pos_dir):
        pos = os.listdir(pos_dir)

    neg_dir = os.path.join(pwd, r'E:\pyitem\SVM\SVM_detectCircle\Lenet\train\Negative1')
    if os.path.exists(neg_dir):
        neg = os.listdir(neg_dir)

    samples0 = []
    samples1 = []
    samples = []
    labels = []

    dx = np.zeros((40, 40))
    dy = np.zeros((40, 40))
    dxy = np.zeros((40, 40))
    t_angle = np.zeros((40, 40))
    for f in pos:
        file_path = os.path.join(pos_dir, f)
        if os.path.exists(file_path):
            pos_img = cv2.imread(file_path)
            pos_img = cv2.resize(pos_img, (40, 40))
            pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2GRAY)
            #求x y方向导数，导数大小+方向
            pos_img_pad = cv2.copyMakeBorder(pos_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            for i in range(1, 41):
                for j in range(1, 41):
                    dx[i-1, j-1] = pos_img_pad[i, j+1] - pos_img_pad[i, j-1]
                    dy[i-1, j-1] = pos_img_pad[i+1, j] - pos_img_pad[i - 1, j]
            dxy = np.sqrt(np.square(dx) + np.square(dy))
            t_angle = dy / (dx+0.0001)
            # descriptors = np.resize(pos_img, (1, 40 * 40 * 3))
            # samples.append(descriptors)
            X1 = np.resize(dxy, (1, 40*40*1))
            # X1 = np.resize(dxy, (40, 40))
            samples0.append(X1)
            X2 = np.resize(t_angle, (1, 40*40*1))
            # X2 = np.resize(t_angle, (40, 40))
            samples1.append(X2)

            labels.append(1.)

    dx1 = np.zeros((40, 40))
    dy1 = np.zeros((40, 40))
    dxy1 = np.zeros((40, 40))
    t_angle1 = np.zeros((40, 40))
    for f in neg:
        file_path = os.path.join(neg_dir, f)
        if os.path.exists(file_path):
            neg_img = cv2.imread(file_path)
            neg_img = cv2.resize(neg_img, (40, 40))
            neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2GRAY)
            # 求x y方向导数，导数大小+方向
            neg_img_pad = cv2.copyMakeBorder(neg_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
            for i in range(1, 41):
                for j in range(1, 41):
                    dx1[i - 1, j - 1] = neg_img_pad[i, j + 1] - neg_img_pad[i, j-1]
                    dy1[i - 1, j - 1] = neg_img_pad[i+1, j] - neg_img_pad[i - 1, j]
            dxy1 = np.sqrt(np.square(dx1) + np.square(dy1))
            t_angle1 = dy1 / (dx1 + 0.0001)
            # descriptors = np.resize(neg_img, (1, 40 * 40 * 3))
            # samples.append(descriptors)
            X11 = np.resize(dxy1, (1, 40*40*1))
            # X11 = np.resize(dxy1, (40, 40 ))
            samples0.append(X11)
            X21 = np.resize(t_angle1, (1, 40*40*1))
            # X21 = np.resize(t_angle1, (40, 40 ))
            samples1.append(X21)

            labels.append(-1.)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    heng = np.arange(0, 40)
    # shu = np.arange(0, 40)
    # ax.plot(samples0[0], samples0[1])
    # print(heng)
    # print(samples0[0][1].shape)
    # plt.scatter(samples0[0][0], samples0[0][1])
    # plt.show()
    # print(samples0[0].shape)
    samples.append(samples0)
    samples.append(samples1)
    samples_number = len(samples0) + len(samples1)

    samples = np.float32(samples)
    samples = np.resize(samples, (samples_number, 40 * 40 * 1))

    labels = np.int32(labels)
    labels = np.resize(labels, (samples_number, 1))

    return samples, labels


def train_svm(samples, labels):
    svm = cv2.ml.SVM_create()

    svm.setKernel(cv2.ml.SVM_POLY)
    # svm.setKernel(cv2.ml.SVM_SIGMOID)
    svm.setType(cv2.ml.SVM_EPS_SVR)

    svm.setP(0.1)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50000, 1e-4)
    svm.setTermCriteria(criteria)

    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)

    wT = svm.getSupportVectors()

    rho, _, _ = svm.getDecisionFunction(0)
    b = -rho

    return wT, b


if __name__ == '__main__':
    samples, labels = load_data()
    wT, b = train_svm(samples, labels)

    img = cv2.imread(r'E:\pyitem\opencv_img\steelPipe-detection\roi_correct.jpg')   # roi_correct.jpg   E:\pyitem\opencv_img\photo\num_circle.jpg

    # for i in range(img.shape[0] // 40):
    #     for j in range(img.shape[1] // 40):
    #
    #         img2 = img.copy()
    #         img1 = img2[i * 40:i * 40 + 40, j * 40:j * 40 + 40]  # 依次遍历32*32的区域
    #         # img1 = img2[i * 16:i * 16 + 32, j * 16:j * 16 + 32]  # 依次遍历32*32的区域
    #         x = np.resize(img1, (40 * 40 * 3, 1))
    #         value = np.dot(wT, x)[0][0] + b
    #         if value > 0:
    #             cv2.circle(img, (i * 40 + 25, j * 40 + 25), 15, (0, 0, 255), 2)
    #             cv2.circle(img, (i * 40 + 25, j * 40 + 25), 1, (0, 255, 255), 2)
    #             # cv2.circle(img, (i * 16 + 16, j * 16 + 16), 16, (0, 0, 255), 2)
    #             # cv2.circle(img, (i * 16 + 16, j * 16 + 16), 1, (0, 255, 255), 2)
    #         # else:
    #         #     cv2.circle(img, (i * 32 + 16, j * 32 + 16), 16, (0, 255, 0), 2)
    #         #     cv2.circle(img, (i*32+16,j*32+16), 1, (0,0,255),2)
    #             # cv2.circle(img, (i * 16 + 16, j * 16 + 16), 1, (0, 0, 255), 2)

    value_los = np.zeros((img.shape[0], img.shape[1]))
    for i in range(20, img.shape[0]-10, 20):
        for j in range(45, img.shape[1]-30, 20):
            # print(i, j)
            img2 = img.copy()
            img1 = img2[i-20:i + 20, j-20:j + 20]  # 依次遍历32*32的区域
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            x = np.resize(img1, (40 * 40 * 1, 1))
            value = np.dot(wT, x)[0][0] + b
            value_los[i][j] = value
            # if value_los[i, j] > 0 and value_los[i, j]>value_los[i-20, j-20]:
            if value>0:
                # cv2.circle(img, (j, i), 15, (0, 0, 255), 2)
                cv2.circle(img, (j, i), 1, (0, 255, 255), 2)
    # for i in range(20, img.shape[0]-20, 15):
    #     for j in range(20, img.shape[1]-20, 15):
    #         if (value_los[i, j]>0 and value_los[i, j]>value_los[i+20,j+20] and value_los[i, j]>value_los[i-20,j-20]):   #and value_los[i, j]>value_los[i+20,j+20]  value_los[i, j]>value_los[i-20,j-20]
    #             value_los[i, j] = value_los[i, j]
    #             # cv2.circle(img, (j, i), 15, (0, 0, 255), 2)
    #             cv2.circle(img, (j, i), 1, (0, 255, 255), 2)
    #         else:
    #             value_los[i, j] = 0


    cv2.imshow('res', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()















