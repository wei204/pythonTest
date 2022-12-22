import cv2
import numpy as np


def watershed_img(img):
    # 平滑色彩
    img_smooth = cv2.pyrMeanShiftFiltering(img, 20, 50)  # sp，定义的漂移物理空间半径大小；sr，定义的漂移色彩空间半径大小

    # 转换为灰度二值图
    gray = cv2.cvtColor(img_smooth, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, gray.max()*0.90, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('binary', binary)
    # ret1, binary1 = cv2.threshold(gray, binary.max()*0.4, 255, cv2.THRESH_BINARY)
    # print(ret1)
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 5)
    # cv2.imshow('binary1', binary1)
    # # 开操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # open_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('open_binary', open_binary)
    # 腐蚀
    erode_close = cv2.erode(binary, kernel)
    # cv2.imshow('erode_open', erode_close)
    # 距离变换
    res = cv2.distanceTransform(binary, cv2.DIST_L2, 5)   # 距离计算方式为欧几里得，掩膜大小3*3
    # cv2.imshow('res', res)
    # 归一化
    res_out = cv2.normalize(res, 0, 1.0, cv2.NORM_MINMAX)   # 归一化范围0-1之间，方式为线性归一化
    cv2.imshow('res_out', res_out*100)
    res_improve = res_out*100
    # 开操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    open_binary = cv2.morphologyEx(res_improve, cv2.MORPH_OPEN, kernel, iterations=3)
    cv2.imshow('open_binary', open_binary)
    # 获取亮点
    ret2, binary2 = cv2.threshold(open_binary, open_binary.max() * 0.40, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary2', binary2)
    binary2_u8 = np.uint8(binary2)
    uk = cv2.subtract(erode_close, binary2_u8)
    ret3, markers = cv2.connectedComponents(binary2_u8)
    print(ret3)
    # 分水岭变换
    markers = markers + 1
    markers[uk==255] = 0
    markers = cv2.watershed(img, markers=markers)
    img[markers==-1] = [0, 0, 255]
    cv2.imshow('result', img)


if __name__ == '__main__':
    img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\num_circle.jpg')
    img = cv2.resize(img, (300, 400))
    cv2.imshow('original', img)
    watershed_img(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()