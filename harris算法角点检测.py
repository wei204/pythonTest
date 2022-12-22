# Open 中的函数 cv2.cornerHarris() 可以用来进行角点检测。参数如
# 下:
# 　　• img - 数据类型为 float32 的输入图像。
# 　　• blockSize - 角点检测中要考虑的领域大小。
# 　　• ksize - Sobel 求导中使用的窗口大小
# 　　• k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06]
import cv2
import numpy as np


#非极大值抑制

def find_max(img1):
    max = np.zeros((img1.shape[0], img1.shape[1]))
    # print(max.shape)
    for i in range(1, img1.shape[0] - 1):
        for j in range(1, img1.shape[1] - 1):
            # if (img1[i, j] > 0.1 * img1.max()):
            #     cv2.circle(img, [j, i], 2, [0, 0, 255], 2)
            roi = img1[i - 1:i + 2, j - 1:j + 2]
            # # print(roi)
            max[i, j] = roi.max()
            return max
            # # max_index = img1[max]
            # if(max > 0.1 * img1.max()):
            #     # img[img1==max] = [0, 0, 255]
            #     cv2.circle(img, )


if __name__ == '__main__':
    # 读入图像
    img = cv2.imread('./photo/qipan1.jpg')
    # print(img.shape)
    img = cv2.resize(img, (614, 316))
    # img_copy = img
    # 转为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.dtype)               # uint8
    # 将输入数据类型转换为float32类型
    gray = np.float32(gray)
    # print(gray.dtype)                  # float32
    # 经过Harris算法处理后图像
    img1 = cv2.cornerHarris(gray, 3, 3, 0.04)  # 返回值为R  R = det(M) - k*(trace(M))^2
    # print(img1.shape)

    # 标记角点,将原图中角点标为黄点
    # img[img1 > 0.1*img1.max()] = [0,255,255]
    # img_copy[img1 > 0.1 * img1.max()] = [0, 0, 255]
    # cv2.imshow('copy', img_copy)

    max = find_max(img1)
    # print(max)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (img1[i, j] == max[i, j]):
                cv2.circle(img, [j, i], 2, [0, 0, 255], 2)
    #
    # # 显示图像
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
