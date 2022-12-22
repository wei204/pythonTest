# import cv2
# import matplotlib .pyplot as plt
# import numpy as np
# img1 = cv2.imread('E:\\pyitem\\opencv_img\\photo\\canon.jpg')
# # img1 = cv2.resize(img1,(300,300))
#
# # 高斯模糊对图像进行平滑处理
# img1_blur = cv2.GaussianBlur(img1, (3, 3), 0)
#
# # print(img1.shape)
# # 窗口显示函数
# def cv2_show(name,img):
#     cv2.imshow(name,img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # 设置双阈值
# v1 = cv2.Canny(img1,18,50)        # 大于上边界一定为边界，小于下边界一定不是边界。因此上边界越大代表条件越严格
# v2 = cv2.Canny(img1_blur,18,50)
#
# # res = np.hstack((v1,v2))
# cv2.imshow('img1',v1)
# cv2.imshow('img1_blur',v2)
# cv2.imshow('img_original', img1)
# cv2.imshow('img_blur_orl', img1_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 窗口显示
# # plt.subplot(111),plt.imshow(v1)
# # plt.subplot(212),plt.imshow(v2)
# # plt.show()
# # 保存图片
# # cv2.imwrite('me1.jpg', v1)


import cv2
import matplotlib.pyplot as plt
from show import img_show
import numpy as np

img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\canon.jpg')
# opencv自带的carry函数
# img_carry = cv2.Canny(img, 20, 150)
# img_show(img_carry)


# 自己写的carry函数 （有点问题）
# 对图像进行模糊处理
img_blur = cv2.GaussianBlur(img, (3, 3), 2)
# 计算梯度
gradx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
grady = cv2.Sobel(img, cv2.CV_64F, 0, 1)
grad_r = np.abs(gradx) + np.abs(grady)  # 梯度的幅值
grad_t = np.arctan(grady/(gradx+1e-3))  # 梯度的方向
print(grad_t)
# # 边缘细化 非极大值抑制
# h, w, c = img.shape
# img_thin = np.zeros((h, w)) # 用于存储细化后的边缘
# for i in range(1, h-1):
#     for j in range(1, w-1):
#         if -np.pi/8<=grad_t[i, j].all()<np.pi/8:
#             if grad_r[i, j] == max([grad_r[i, j-1].all(), grad_r[i, j].all(), grad_r[i, j+1].all()]):
#                 img_thin[i, j] = grad_r[i, j]
#         elif np.pi/8<=grad_t[i, j].all()<3*np.pi/8:
#             if grad_r[i, j] == max([grad_r[i, j-1].all(), grad_r[i, j].all(), grad_r[i, j+1].all()]):
#                 img_thin[i, j] = grad_r[i, j]
#         elif 3*np.pi/8<=grad_t[i, j].all()<5*np.pi/8:
#             if grad_r[i, j] == max([grad_r[i, j-1].all(), grad_r[i, j].all(), grad_r[i, j+1].all()]):
#                 img_thin[i, j] = grad_r[i, j]
#         elif 5*np.pi/8<=grad_t[i, j].all()<7*np.pi/8:
#             if grad_r[i, j] == max([grad_r[i, j-1].all(), grad_r[i, j].all(), grad_r[i, j+1].all()]):
#                 img_thin[i, j] = grad_r[i, j]
# img_show(img_thin)
# # 双阈值抑制
# tha = 20
# thb = 150
# img_edge = np.zeros((h, w)) # 用于存储最终边缘
# for i in range(1, h-1):
#     for j in range(1, w-1):
#         if img_thin[i, j] >= thb:
#