import cv2
import numpy as np


def line_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 110)  # 返回值lines为极坐标点阵,函数自动筛选重合点最多的点
    for line in lines:
        print(line, type(line))
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*a)
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('line_detection', img)


def lineP_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=20)  # 返回值lines为极坐标点阵
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('lineP_detect', img)


# 检测圆
def circle_detection(img):
    # 对图像进行色彩平滑
    filt = cv2.pyrMeanShiftFiltering(img, 50, 100)   # 关键参数是sp和sr的设置，二者设置的值越大，
                                                       # 对图像色彩的平滑效果越明显，同时函数耗时也越多
    # filt = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imshow('filter', filt)
    gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)  # param2 值越小，检测到的元越多
                                                          # 返回值为圆心坐标及半径
    circles = np.uint16(np.around(circles))
    # print(circles)
    for i in circles[0]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv2.circle(img, (i[0], i[1]), 1, (0, 255, 255), 2)
    cv2.imshow('circle_detection', img)




if __name__ == '__main__':
    img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\captcha.jpg')
    # lineP_detect(img)
    line_detection(img)
    # circle_detection(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()