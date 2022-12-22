# import  cv2
# import numpy as np
#
#
# # 定义sobel算子计算梯度，边缘检测
# def sobel_img(img):
#     # sobel 算子
#     grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     # scharr 算子  效果更好
#     # grad_x = cv2.Scharr(img, cv2.CV_32F, 1, 0)
#     # grad_y = cv2.Scharr(img, cv2.CV_32F, 0, 1)
#     gradx = cv2.convertScaleAbs(grad_x)    # 先计算数组绝对值，后转化为8位无符号数
#     grady = cv2.convertScaleAbs(grad_y)
#     cv2.imshow('gradient_x', gradx)
#     cv2.imshow('gradient_y', grady)
#
#     gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
#     cv2.imshow('gradient', gradxy)
#     # 平滑处理
#     gradxy_blur = cv2.GaussianBlur(gradxy, (3, 3), 0)
#     cv2.imshow('gradient_blur', gradxy_blur)
#
#
# # 定义拉普拉斯方式边缘检测
# def lapalian_img(img):
#     # res = cv2.Laplacian(img, cv2.CV_32F)
#     # lpls = cv2.convertScaleAbs(res)
#     # cv2.imshow('lapalian_img', lpls)
#
#     # 自定义算子，边缘检测
#     kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
#     res = cv2.filter2D(img, cv2.CV_32F, kernel=kernel)
#     lpls = cv2.convertScaleAbs(res)
#     cv2.imshow('lapalian_free', lpls)
#
#
# if __name__ == '__main__':
#     img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\canon.jpg')
#     sobel_img(img)
#     # lapalian_img(img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# #
#--------------------------------------手写版-----------------------------------------
import cv2
import numpy as np
import math
#
# def non_maximum_suppression(dx_gray, dy_gray, df_gray):
#     '''
#     dx_gray:x方向梯度矩阵
#     dy_gray:y方向梯度矩阵
#     df_gray:梯度强度矩阵
#     '''
#     df_gray = np.pad(df_gray, ((1, 1), (1, 1)), constant_values=0)  # 填充
#     h, w = df_gray.shape
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             if df_gray[i, j] != 0:
#                 # gx = math.fabs(dx_gray[i - 1, j - 1])
#                 # gy = math.fabs(dy_gray[i - 1, j - 1])
#                 # if gx > gy:
#                 #     weight = gy / gx
#                 #     grad1 = df_gray[i + 1, j]
#                 #     grad2 = df_gray[i - 1, j]
#                 #     if gx * gy > 0:
#                 #         grad3 = df_gray[i + 1, j + 1]
#                 #         grad4 = df_gray[i - 1, j - 1]
#                 #     else:
#                 #         grad3 = df_gray[i + 1, j - 1]
#                 #         grad4 = df_gray[i - 1, j + 1]
#                 # else:
#                 #     weight = gx / gy
#                 #     grad1 = df_gray[i, j + 1]
#                 #     grad2 = df_gray[i, j - 1]
#                 #     if gx * gy > 0:
#                 #         grad3 = df_gray[i + 1, j + 1]
#                 #         grad4 = df_gray[i - 1, j - 1]
#                 #     else:
#                 #         grad3 = df_gray[i + 1, j - 1]
#                 #         grad4 = df_gray[i - 1, j + 1]
#                 # t1 = weight * grad1 + (1 - weight) * grad3
#                 # t2 = weight * grad2 + (1 - weight) * grad4
#
#
#                 gx = dx_gray[i - 1, j - 1]
#                 gy = dy_gray[i - 1, j - 1]
#                 if gx*gy>0:
#                     gx = math.fabs(gx)
#                     gy = math.fabs(gy)
#                     if(gx>gy):
#                         weight = gy / gx
#                         grad1 = df_gray[i, j-1]
#                         grad2 = df_gray[i, j+1]
#                     else:
#                         weight = gx / gy
#                         grad1 = df_gray[i-1, j]
#                         grad2 = df_gray[i+1, j]
#                     grad3 = df_gray[i - 1, j - 1]
#                     grad4 = df_gray[i + 1, j + 1]
#                 else:
#                     gx = math.fabs(gx)
#                     gy = math.fabs(gy)
#                     if(gx>gy):
#                         weight = gy / gx
#                         grad1 = df_gray[i, j-1]
#                         grad2 = df_gray[i, j+1]
#                         grad3 = df_gray[i+1, j-1]
#                         grad4 = df_gray[i-1, j+1]
#                     else:
#                         weight = gx / gy
#                         grad1 = df_gray[i - 1, j]
#                         grad2 = df_gray[i + 1, j]
#                         grad3 = df_gray[i-1, j+1]
#                         grad4 = df_gray[i+1, j-1]
#                 t1 = (1-weight) * grad1 + weight * grad3
#                 t2 = (1-weight) * grad2 + weight * grad4
#                 if df_gray[i, j] > t1 and df_gray[i, j] > t2:
#                     df_gray[i, j] = df_gray[i, j]
#                 else:
#                     df_gray[i, j] = 0
#     return df_gray
#
img = cv2.imread('photo/canon.jpg')

#计算x轴（水平）、y轴（竖直）方向梯度
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
img_x = img.copy()
img_y = img.copy()
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        img_x[i, j] = img[i+1, j] - img[i, j]   #y方向梯度 显示水平边缘
        img_y[i, j] = img[i, j+1] - img[i, j]    #x方向梯度 显示竖直边缘
img_x = img_x[1:img_x.shape[0]-1, 1:img_x.shape[1]-1]
img_y = img_y[1:img_y.shape[0]-1, 1:img_y.shape[1]-1]
cv2.imshow('x', img_x)
cv2.imshow('y', img_y)
print(img_y.shape)
#
# # 利用sobel算子计算梯度
# #灰度化
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #高斯平滑
# img_G = cv2.GaussianBlur(gray, (3, 3), 0)
# img = cv2.copyMakeBorder(img_G, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
# img_x = np.zeros((img.shape[0], img.shape[1]), np.uint8)
# img_y = np.zeros((img.shape[0], img.shape[1]), np.uint8)
# # img_xy = np.zeros((img.shape[0], img.shape[1]), np.uint8)
# dst = np.zeros((img.shape[0], img.shape[1]), np.uint8)
# for i in range(1, img.shape[0]-1):
#     for j in range(1, img.shape[1]-1):
#         img_x = img[i+1, j-1]+2*img[i+1, j]+img[i+1, j+1]-img[i-1, j-1]-2*img[i-1, j]-img[i-1, j+1]
#         img_y = img[i - 1, j + 1] + 2 * img[i, j+1] + img[i + 1, j + 1] - img[i - 1, j - 1] - 2 * img[
#             i, j-1] - img[i+1, j -1]
#         img_xy = np.sqrt(img_x**2+img_y**2)
#         if img_xy>20:
#             dst[i, j] = 255
#         else:
#             dst[i, j] = 0
#
# dst = dst[1:dst.shape[0]-1, 1:dst.shape[1]-1]
# cv2.imshow('dst', dst)
# print(dst.shape)
#
#
cv2.waitKey(0)
cv2.destroyAllWindows()



# #----------------------------------参考版-----------------------------------------
# import cv2
# import numpy as np
# import math
#
# def partial_derivative(new_gray_img):
#     '''
#     new_gray_img:高斯卷积后的灰度图
#     '''
#     new_gray_img = np.pad(new_gray_img, ((0, 1), (0, 1)), constant_values=0)  # 填充
#     h, w = new_gray_img.shape
#     dx_gray = np.zeros([h - 1, w - 1])  # 用来存储x方向偏导
#     dy_gray = np.zeros([h - 1, w - 1])  # 用来存储y方向偏导
#     # df_gray = np.zeros([h - 1, w - 1])  # 用来存储梯度强度
#     for i in range(h - 1):
#         for j in range(w - 1):
#             dx_gray[i, j] = new_gray_img[i, j + 1] - new_gray_img[i, j]
#             dy_gray[i, j] = new_gray_img[i + 1, j] - new_gray_img[i, j]
#             # df_gray[i, j] = np.sqrt(np.square(dx_gray[i, j]) + np.square(dy_gray[i, j]))
#     return dx_gray, dy_gray
#
# def sobel_dst(new_gray_img):
#     img = cv2.copyMakeBorder(new_gray_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
#     dst = np.zeros((img.shape[0], img.shape[1]), np.uint8)
#     for i in range(1, img.shape[0]-1):
#         for j in range(1, img.shape[1]-1):
#             img_x = img[i+1, j-1]+2*img[i+1, j]+img[i+1, j+1]-img[i-1, j-1]-2*img[i-1, j]-img[i-1, j+1]
#             img_y = img[i - 1, j + 1] + 2 * img[i, j+1] + img[i + 1, j + 1] - img[i - 1, j - 1] - 2 * img[
#                 i, j-1] - img[i+1, j -1]
#             img_xy = np.sqrt(img_x**2+img_y**2)
#             if img_xy>20:
#                 dst[i, j] = 255
#             else:
#                 dst[i, j] = 0
#     dst = dst[1:dst.shape[0]-1, 1:dst.shape[1]-1]
#     return dst
#
# # 非极大值抑制
# def non_maximum_suppression(dx_gray, dy_gray, df_gray):
#     '''
#     dx_gray:x方向梯度矩阵
#     dy_gray:y方向梯度矩阵
#     df_gray:梯度强度矩阵
#     '''
#     df_gray = np.pad(df_gray, ((1, 1), (1, 1)), constant_values=0)  # 填充
#     df_gray = np.uint8(df_gray)
#     h, w = df_gray.shape
#     for i in range(1, h - 1):
#         for j in range(1, w - 1):
#             if df_gray[i, j] != 0:
#                 gx = dx_gray[i - 1, j - 1]
#                 gy = dy_gray[i - 1, j - 1]
#                 if gx*gy>0:
#                     gx = math.fabs(gx)
#                     gy = math.fabs(gy)
#                     if(gx>gy):
#                         weight = gy / gx+1
#                         grad1 = df_gray[i, j-1]
#                         grad2 = df_gray[i, j+1]
#                     else:
#                         weight = gx / gy+1
#                         grad1 = df_gray[i-1, j]
#                         grad2 = df_gray[i+1, j]
#                     grad3 = df_gray[i - 1, j - 1]
#                     grad4 = df_gray[i + 1, j + 1]
#                 else:
#                     gx = math.fabs(gx)
#                     gy = math.fabs(gy)
#                     if(gx>gy):
#                         weight = gy / gx+1
#                         grad1 = df_gray[i, j-1]
#                         grad2 = df_gray[i, j+1]
#                         grad3 = df_gray[i+1, j-1]
#                         grad4 = df_gray[i-1, j+1]
#                     else:
#                         # weight = gx / gy+1
#                         weight = 0.3
#                         grad1 = df_gray[i - 1, j]
#                         grad2 = df_gray[i + 1, j]
#                         grad3 = df_gray[i-1, j+1]
#                         grad4 = df_gray[i+1, j-1]
#                 t1 = (1-weight) * grad1 + weight * grad3
#                 t2 = (1-weight) * grad2 + weight * grad4
#                 if df_gray[i, j] > t1 and df_gray[i, j] > t2:
#                     df_gray[i, j] = df_gray[i, j]
#                 else:
#                     df_gray[i, j] = 0
#     return df_gray
#
# if __name__ == '__main__':
#     img = cv2.imread('photo/canon.jpg')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     new_gray_img = cv2.GaussianBlur(gray, (3, 3), 0)
#     # 得到sobel边缘图
#     dst = sobel_dst(new_gray_img)
#     cv2.imshow('dst', dst)
#     # 得到x，y方向的梯度大小
#     dx, dy = partial_derivative(new_gray_img)
#     # 非极大值抑制
#     img_max = non_maximum_suppression(dx, dy, dst)
#     cv2.imshow('max', img_max)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()