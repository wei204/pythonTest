import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 生成高斯核（改造，再次改造）
def gaussian_create():
    sigma1 = sigma2 = 1
    gaussian_sum = 0
    g = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            g[i, j] = math.exp(-1 / 2 * (np.square(i - 1) / np.square(sigma1)
                                         + (np.square(j - 1) / np.square(sigma2)))) / (
                              2 * math.pi * sigma1 * sigma2)
            gaussian_sum = gaussian_sum + g[i, j]
    g = g / gaussian_sum  # 归一化
    return g


# 产生灰度图
def gray_fuc(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# 高斯卷积
def gaussian_blur(gray_img, g):
    '''
    gray_img:灰度图
    g:高斯核
    '''
    gray_img = np.pad(gray_img, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = gray_img.shape
    new_gray_img = np.zeros([h - 2, w - 2])
    for i in range(h - 2):
        for j in range(w - 2):
            new_gray_img[i, j] = np.sum(gray_img[i:i + 3, j:j + 3] * g)
    return new_gray_img


# 求高斯偏导
def partial_derivative(new_gray_img):
    '''
    new_gray_img:高斯卷积后的灰度图
    '''
    new_gray_img = np.pad(new_gray_img, ((0, 1), (0, 1)), constant_values=0)  # 填充
    h, w = new_gray_img.shape
    dx_gray = np.zeros([h - 1, w - 1])  # 用来存储x方向偏导
    dy_gray = np.zeros([h - 1, w - 1])  # 用来存储y方向偏导
    df_gray = np.zeros([h - 1, w - 1])  # 用来存储梯度强度
    for i in range(h - 1):
        for j in range(w - 1):
            dx_gray[i, j] = new_gray_img[i, j + 1] - new_gray_img[i, j]
            dy_gray[i, j] = new_gray_img[i + 1, j] - new_gray_img[i, j]
            df_gray[i, j] = np.sqrt(np.square(dx_gray[i, j]) + np.square(dy_gray[i, j]))
    return dx_gray, dy_gray, df_gray


# 非极大值抑制
def non_maximum_suppression(dx_gray, dy_gray, df_gray):
    '''
    dx_gray:x方向梯度矩阵
    dy_gray:y方向梯度矩阵
    df_gray:梯度强度矩阵
    '''
    df_gray = np.pad(df_gray, ((1, 1), (1, 1)), constant_values=0)  # 填充
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] != 0:
                # gx = math.fabs(dx_gray[i - 1, j - 1])
                # gy = math.fabs(dy_gray[i - 1, j - 1])
                # if gx > gy:
                #     weight = gy / gx
                #     grad1 = df_gray[i + 1, j]
                #     grad2 = df_gray[i - 1, j]
                #     if gx * gy > 0:
                #         grad3 = df_gray[i + 1, j + 1]
                #         grad4 = df_gray[i - 1, j - 1]
                #     else:
                #         grad3 = df_gray[i + 1, j - 1]
                #         grad4 = df_gray[i - 1, j + 1]
                # else:
                #     weight = gx / gy
                #     grad1 = df_gray[i, j + 1]
                #     grad2 = df_gray[i, j - 1]
                #     if gx * gy > 0:
                #         grad3 = df_gray[i + 1, j + 1]
                #         grad4 = df_gray[i - 1, j - 1]
                #     else:
                #         grad3 = df_gray[i + 1, j - 1]
                #         grad4 = df_gray[i - 1, j + 1]
                # t1 = weight * grad1 + (1 - weight) * grad3
                # t2 = weight * grad2 + (1 - weight) * grad4


                gx = dx_gray[i - 1, j - 1]
                gy = dy_gray[i - 1, j - 1]
                if gx*gy>0:
                    gx = math.fabs(gx)
                    gy = math.fabs(gy)
                    if(gx>gy):
                        weight = gy / gx
                        grad1 = df_gray[i, j-1]
                        grad2 = df_gray[i, j+1]
                    else:
                        weight = gx / gy
                        grad1 = df_gray[i-1, j]
                        grad2 = df_gray[i+1, j]
                    grad3 = df_gray[i - 1, j - 1]
                    grad4 = df_gray[i + 1, j + 1]
                else:
                    gx = math.fabs(gx)
                    gy = math.fabs(gy)
                    if(gx>gy):
                        weight = gy / gx
                        grad1 = df_gray[i, j-1]
                        grad2 = df_gray[i, j+1]
                        grad3 = df_gray[i+1, j-1]
                        grad4 = df_gray[i-1, j+1]
                    else:
                        weight = gx / gy
                        grad1 = df_gray[i - 1, j]
                        grad2 = df_gray[i + 1, j]
                        grad3 = df_gray[i-1, j+1]
                        grad4 = df_gray[i+1, j-1]
                t1 = (1-weight) * grad1 + weight * grad3
                t2 = (1-weight) * grad2 + weight * grad4
                if df_gray[i, j] > t1 and df_gray[i, j] > t2:
                    df_gray[i, j] = df_gray[i, j]
                else:
                    df_gray[i, j] = 0
    return df_gray


# 双阈值过滤 低于低阈值的置零，高于高阈值的保留，高于低阈值低于高阈值，且八邻域与边缘连接的保留，其他置零
def double_threshold(df_gray, low, high):
    '''
    df_gray:梯度强度矩阵
    low:低阈值
    high:高阈值
    '''
    h, w = df_gray.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if df_gray[i, j] < low:
                df_gray[i, j] = 0
            elif df_gray[i, j] > high:
                df_gray[i, j] = 1
            # elif (df_gray[i, j - 1] > high) or (df_gray[i - 1, j - 1] > high) or (
            #         df_gray[i + 1, j - 1] > high) or (df_gray[i - 1, j] > high) or (df_gray[i + 1, j] > high) or (
            #         df_gray[i - 1, j + 1] > high) or (df_gray[i, j + 1] > high) or (df_gray[i + 1, j + 1] > high):
            #     df_gray[i, j] = 1
            elif ((df_gray[i, j - 1] > high)  or (df_gray[i - 1, j] > high) or (df_gray[i + 1, j] > high) or
                   (df_gray[i, j + 1] > high)):
                df_gray[i, j] = 1
            else:
                df_gray[i, j] = 0
    return df_gray


if __name__ == '__main__':
    # 读取图像
    img = plt.imread(r'E:\pyitem\opencv_img\steelPipe-detection\roi_correct.jpg')
    # cv2.imshow("ori", img)

    # 生成高斯核
    gaussian = gaussian_create()
    # 生成灰度图
    gray = gray_fuc(img)
    # 高斯卷积
    new_gray = gaussian_blur(gray, gaussian)
    # new_gray = new_gray / 255
    # cv2.imshow('gray', new_gray)

    # 求偏导
    # new_gray = gray
    d = partial_derivative(new_gray)
    dx = d[0]
    dy = d[1]
    df = d[2]
    # 非极大值抑制
    new_df = non_maximum_suppression(dx, dy, df)
    # 双阈值过滤,并将图像转换成转化二值图
    # new_df = new_gray/255
    # print(new_df)
    low_threshold = 0.01 * np.max(new_df)
    high_threshold = 0.02 * np.max(new_df)
    result = double_threshold(new_df, low_threshold, high_threshold)

    # 绘制灰度图
    plt.hist(result.ravel(), 1, [0, 1])  # img.ravel() 将多维数组转为一维数组
    plt.show()
    # 输出图像
    plt.imshow(result, cmap="gray")
    plt.axis("off")
    plt.show()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()