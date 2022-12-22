import cv2
import numpy as np

# 图像相加
def add_img(img1, img2):
    res = cv2.add(img1, img2)
    cv2.imshow('add', res)


# 图像相减
def sub_img(img1, img2):
    res = cv2.subtract(img1, img2)
    cv2.imshow('sub', res)


# 图像相乘
def mul_img(img1, img2):
    res = cv2.multiply(img1, img2)
    cv2.imshow('mul', res)


# 图像相除
def div_img(img1, img2):
    res = cv2.divide(img1, img2)
    cv2.imshow('div', res)


# 图像像素相与
def img_and(img1, img2):
    res = cv2.bitwise_and(img1, img2)
    cv2.imshow('and', res)


# 图像像素相或
def img_or(img1, img2):
    res = cv2.bitwise_or(img1, img2)
    cv2.imshow('or', res)

# 图像像素非运算
def img_not(img1):
    res = cv2.bitwise_not(img1)
    cv2.imshow('not', res)


# 图像像素异或
def img_xor(img1, img2):
    res = cv2.bitwise_xor(img1, img2)
    cv2.imshow('xor', res)

# 增强图像对比度
def contrast_brightness(img, c, b):
    h, w, ch = img.shape
    blank = np.zeros([h, w, ch], img.dtype)
    res = cv2.addWeighted(img, c, blank, 1-c, b)     # c 为img所占权重，1-c 为blank所占权重，b为线性增加的值
    cv2.imshow('c_b', res)


# 计算像素均值
def mean_img(img1, img2):
    M1 = cv2.mean(img1)
    M2 = cv2.mean(img2)
    print(M1, M2)


if __name__ == '__main__':
    img1 = cv2.imread('star.jpg')
    img1 = cv2.resize(img1, (300, 300))
    # print(img1.shape)
    img2 = cv2.imread('me1.jpg')
    # print(img2.shape)
    # add_img(img1, img2)
    sub_img(img2, img1)
    # mul_img(img1, img2)
    # div_img(img2, img1)
    # mean_img(img1, img2)
    # img_and(img1, img2)
    # img_or(img1, img2)
    # img_not(img2)
    # img_xor(img1, img2)
    # cv2.imshow('original', img2)
    # contrast_brightness(img2, 1.5, 10)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
