import cv2
from matplotlib import pyplot as plt
import numpy as np

# 全局二值化
def binaryzation_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.hist(gray.ravel(), 256, [0, 256])  # img.ravel() 将多维数组转为一维数组
    # plt.show()
    # cv2.imshow('gray', gray)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 大于阈值时为白色，小于阈值为黑色
    print(ret)   # 显示阈值
    # plt.hist(binary.ravel(), 256, [0, 256])  # img.ravel() 将多维数组转为一维数组
    # plt.show()
    cv2.imshow('binaryzation_img', binary)


# 局部阈值二值化
def local_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)  # 25代表5*5核，10代表阈值常数项
    cv2.imshow('local_binary', binary)


# 大图像二值化   全局二值化
def big_img_binary():
    img = cv2.imread('C:\\Users\\Administrator\\Desktop\\user\\information\\0-0.jpg')
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(0, h, 10):
        for col in range(0, w, 20):
            roi = gray[row:row+10, col:col+20]      # 每次二值化处理时所选的区域大小
            ret, res = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   # 二值化处理
            # gray[row:row + 3, col:col + 5] = res
            print(np.std(res), np.mean(res))
            if np.mean(res) > 200:
                gray[row:row + 10, col:col + 20] = 255
            else:
                gray[row:row + 10, col:col + 20] = res   # 将处理后的值返回原位置
    cv2.imshow('big_img_binary', gray)
    # cv2.imwrite('C:\\Users\\Administrator\\Desktop\\user\\information\\me_binary.jpg', gray)


# 大图像二值化   局部二值化
def big_img_local():
    img = cv2.imread('C:\\Users\\Administrator\\Desktop\\user\\information\\0-0.jpg')
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for row in range(0, h, 10):
        for col in range(0, w, 10):
            roi = gray[row:row+10, col:col+10]      # 每次二值化处理时所选的区域大小
            res = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5)   # 二值化处理
                               # 常数越大，阈值越低，二值化处理后图像越白    第二个参数为满足阈值条件的像素点处理后的灰度值
            gray[row:row + 10, col:col + 10] = res   # 将处理后的值返回原位置
    cv2.imshow('big_img_binary', gray)
    # cv2.imwrite('C:\\Users\\Administrator\\Desktop\\user\\information\\me_local.jpg', gray)



if __name__ == '__main__':

    # img = cv2.resize(img, (352, 500))
    # binaryzation_img(img)
    # local_binary(img)
    big_img_binary()
    # big_img_local()
    cv2.waitKey(0)
    cv2.destroyAllWindows()