import cv2
from matplotlib import pyplot as plt
import numpy as np

# 显示数据直方图
def img_plot(img):
    plt.hist(img.ravel(), 256, [0, 256])    # img.ravel() 将多维数组转为一维数组
    plt.show()


# 绘制图像直方图
def img_hist(img):
    # 定义三通道灰度曲线颜色
    color = ('blue', 'green', 'red')
    # 分别绘制三通道灰度直方图
    for i, color in enumerate(color):        # enumerate 函数返回值为索引+值
        hist = cv2.calcHist(img, [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])          # 定义x轴最小值，最大值
    plt.show()

# 直方图均衡化，只能处理灰度图
def equalHist_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    return hist
    # cv2.imshow('equalHist', hist)


# 直方图均衡化，分通道处理
def equalHist1_img(img):
    b, g, r = cv2.split(img)
    b1 = cv2.equalizeHist(b)
    g1 = cv2.equalizeHist(g)
    r1 = cv2.equalizeHist(r)
    hist = cv2.merge([b1, g1, r1])
    # return hist
    cv2.imshow('equal1Hist', hist)


# 局部直方图均衡化
def local_equalhist(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.createCLAHE(5, (7, 7))      #对比度参数为5，每次处理的区域为7*7
    res = hist.apply(gray)
    # print(res.shape)
    return res
    # cv2.imshow('local_equalhist', res)


if __name__ == '__main__':
    img1 = cv2.imread('E:\\pyitem\\opencv_img\\photo\\canon.jpg')
    print(img1)
    # print(img1.shape)


    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # b, g, r = cv2.split(img1)
    # print(b.shape, g.shape, r.shape)
    # img_plot(img1)
    # img_hist(img1)
    # print(img1)    # 输出灰度矩阵
    # cv2.imshow('original', img1)

    # img0 = equalHist_img(img1)
    # # equalHist1_img(img1)
    # img2 = local_equalhist(img1)
    # img_plot(img1)
    # img_plot(img0)
    # img_plot(img2)
    # img2 = equalHist1_img(img1)
    # img_plot(img2)
    # equalHist1_img(img1)


    # n = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    # m = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # # print(n[2:7, 2:7])
    # # print(n[4, 4])
    # for x in range(1, n.shape[0]-1):
    #     for y in range(1, n.shape[1]-1):
    #         roi = n[x-1:x+2, y-1:y+2]
    #         # print(roi)
    #         mean = np.sum(roi*m)/8
    #         mean1 = (n[x-1][y-1]+n[x][y-1]+n[x+1][y-1]
    #                                 +n[x-1][y]+n[x+1][y]+n[x-1][y+1]
    #                                 +n[x][y+1]+n[x+1][y+1])/8
    #         print(mean, mean1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()