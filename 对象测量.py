import cv2
import numpy as np
from matplotlib import pyplot as plt


def measure_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
    cv2.imshow('binary', binary)
    # # 直方图
    # plt.hist(binary.ravel(), 256, [0, 256])
    # plt.show()
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # 计算图形轮廓面积
        area = cv2.contourArea(contour)
        # 轮廓外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w*h    # 外接矩形面积
        # 绘制轮廓
        # image1 = image.copy()
        cv2.drawContours(image, contours, i, (0, 255, 255), 2)
        # 图像质心
        mm = cv2.moments(contour)
        print(type(mm))   # 类型为字典
        cx = mm['m10']/mm['m00']
        cy = mm['m01']/mm['m00']
        cv2.circle(image, (np.int16(cx), np.int16(cy)), 3, (255, 0, 0), -1)  # 绘制质心
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print(area, rect_area)
        print(float(area)/float(rect_area))     # 轮廓面积与外接矩形面积之比
    # cv2.imshow('contour', image)
    cv2.imshow('measure_object', image)


img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\wzm.jpg')
measure_object(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
