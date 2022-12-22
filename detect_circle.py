import cv2
import numpy as np


# 检测圆
def circle_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)  # param2 值越小，检测到的元越多
    #                                                       # 返回值为圆心坐标及半径
    # circles = np.uint16(np.around(circles))
    # print(circles)
    # for i in circles[0]:
    #     cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
    #     cv2.circle(img, (i[0], i[1]), 1, (0, 255, 255), 2)
    # cv2.imshow('circle_detection', img)

    #在边缘二值图上检测圆
    edges = cv2.Canny(gray, 10, 100)
    cv2.imwrite('photo/circle_edges.jpg', edges)
    edges0 = cv2.imread('photo/circle_edges.jpg')
    gray0 = cv2.cvtColor(edges0, cv2.COLOR_BGR2GRAY)
    # print(edges0.shape)
    # # cv2.imshow('edges', edges)

    #直接在原灰度图上检测圆
    #num_circle 参数
    # circles = cv2.HoughCircles(gray0, cv2.HOUGH_GRADIENT, 1, 15, param1=150, param2=10, minRadius=10, maxRadius=18)
    circles = cv2.HoughCircles(gray0, cv2.HOUGH_GRADIENT, 1, 15, param1=150, param2=10, minRadius=3, maxRadius=10)
    circles = np.uint16(np.around(circles))
    print('圆的个数为：', len(circles[0]))
    for i in circles[0]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)   #(i[0], i[1])代表圆心，i[2]代表半径
        cv2.circle(img, (i[0], i[1]), 1, (0, 255, 255), 2)
    cv2.imshow('circle_detection', img)

if __name__ == '__main__':
    img = cv2.imread('photo/circle-aug.jpg')
    circle_detection(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

