import cv2
import numpy as np


img = cv2.imread(r'E:\pyitem\opencv_img\photo\num_circle.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
max_pos = 0
min_pos = 0

def upmax_pos(x):
    global max_pos, min_pos, gray, circles
    max_pos = cv2.getTrackbarPos('max_pos', 'window')
    min_pos = cv2.getTrackbarPos('min_pos', 'window')
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 15, max_pos, min_pos, 10, 18)
    circles = np.uint16(np.around(circles))
    for i in circles[0]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)   #(i[0], i[1])代表圆心，i[2]代表半径
        cv2.circle(img, (i[0], i[1]), 1, (0, 255, 255), 2)


cv2.namedWindow('window')
cv2.createTrackbar('max_pos', 'window', max_pos, 255, upmax_pos)
cv2.createTrackbar('min_pos', 'window', min_pos, 255, upmax_pos)
cv2.setTrackbarPos('max_pos', 'window', 150)
cv2.setTrackbarPos('min_pos', 'window', 15)
while(1):

    cv2.imshow('window', img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# alpha = 0.3
# beta = 80
# img_path = r"E:\pyitem\opencv_img\photo\num_circle.jpg"
# img = cv2.imread(img_path)
# img2 = cv2.imread(img_path)
# def updateAlpha(x):
#     global alpha, img, img2
#     # 得到数值
#     alpha = cv2.getTrackbarPos('Alpha', 'image')
#     alpha = alpha * 0.01
#     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
# def updateBeta(x):
#     global beta, img, img2
#     beta = cv2.getTrackbarPos('Beta', 'image')
#     img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))
# # 创建窗口
# cv2.namedWindow('image')
# cv2.createTrackbar('Alpha', 'image', 0, 300, updateAlpha)
# cv2.createTrackbar('Beta', 'image', 0, 255, updateBeta)
# # 设置默认值
# cv2.setTrackbarPos('Alpha', 'image', 100)
# cv2.setTrackbarPos('Beta', 'image', 10)
# while (True):
#     cv2.imshow('image', img)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cv2.destroyAllWindows()


