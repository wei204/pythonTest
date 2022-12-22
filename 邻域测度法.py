import matplotlib.pyplot as plt
import numpy as np
import cv2


# def neighbor_meansure(img, k):
#     # 对原图边缘填充
#     img1 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
#     img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     # 遍历图像每个像素
#     for x in range(1, img_gray.shape[0]-1):
#         for y in range(1, img_gray.shape[1]-1):
#             # 取出roi区域
#             roi = img_gray[x-1:x+1+1, y-1:y+1+1]
#             # print(roi)
#             # 计算roi区域灰度平均值
#             mean = np.sum(roi*n)/8
#             # 计算修正后的各个像素点灰度值
#             img_gray[x][y] = img_gray[x][y] + k*(img_gray[x][y]-mean)
#     # 转为彩色图
#     img2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
#     return img2

def neighbor_meansure(img, k):
    # 对原图边缘填充
    img1 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    h, w, c = img1.shape
    print('img1', img1.shape)
    # img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # 遍历图像每个像素
    for x in range(1, img1.shape[0]-1):
        for y in range(1, img1.shape[1]-1):
            # 取出roi区域
            roi = img1[x-1:x+1+1, y-1:y+1+1]
            # print(roi)
            # 计算roi区域灰度平均值
            mean = np.sum(roi*n)/8
            # 计算修正后的各个像素点灰度值
            img1[x][y] = img1[x][y] + k*(img1[x][y]-mean)
    # # 转为彩色图
    # img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 裁剪为原图大小
    img2 = img1[1:h-1, 1:w-1]
    return img2


if __name__ == '__main__':
    # 定义卷积模板
    n = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    # 输入图像
    img = cv2.imread('./photo/pyramid.jpg')
    print(img.shape)
    cv2.imshow('original', img)
    # # 显示原图直方图
    # plt.hist(img.ravel(), 256, [0, 255])
    # plt.show()
    # # 经过不同k值的邻域测度处理
    # img_improve0_2 = neighbor_meansure(img, 0.2)
    img_improve0_8 = neighbor_meansure(img, 0.8)
    print(img_improve0_8.shape)
    # img_improve1_5 = neighbor_meansure(img, 1.5)
    # # 显示处理后图像
    # cv2.imshow('improve0_2', img_improve0_2)
    cv2.imshow('improve0_8', img_improve0_8)
    # cv2.imshow('improve1_5', img_improve1_5)
    # # 显示处理后的灰度直方图
    # plt.hist(img_improve0_2.ravel(), 256, [0, 255])
    # plt.show()
    # plt.hist(img_improve0_8.ravel(), 256, [0, 255])
    # plt.show()
    # plt.hist(img_improve1_5.ravel(), 256, [0, 255])
    # plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



