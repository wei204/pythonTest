import cv2
import numpy as np
import matplotlib.pyplot as plt

# def hist(img):
#     hist = np.zeros((255, 1))
#     for h in range(img.shape[0]):
#         for w in range(img.shape[1]):
#             hist[img[h, w]] += 1
#     return hist
#
#
# if __name__ == '__main__':
#     # img = cv2.imread('./photo/canon.jpg')
#     # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # hist = hist(gray)
#     # x = np.zeros((255, 1))
#     # for i in range(255):
#     #     x[i] = i
#     #
#     # plt.figure()
#     # plt.plot(x, hist)
#     # plt.show()
#
#     img = cv2.imread('./photo/2_28.jpg')
#     # cv2.imshow('in', img)
#     # img = cv2.resize(img, (28, 28))
#     # print(img.shape)
#     img = cv2.bitwise_not(img)
#     # cv2.imshow('out',img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     cv2.imwrite('./photo/2_28.jpg', img)


##添加进度条
# img = cv2.imread(r'E:\pyitem\opencv_img\photo\shuzhuo.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('Canny')
# def nothing(x):
#     pass
#
# cv2.createTrackbar('threshold1', 'Canny', 10, 100, nothing)
# cv2.createTrackbar('threshold2', 'Canny', 50, 300, nothing)
# while(1):
#     threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
#     threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
#     img_edges = cv2.Canny(gray, threshold1, threshold2)
#     cv2.imshow('Canny', img_edges)
#     if cv2.waitKey(1)==ord('q'):
#         break
#
# cv2.destroyAllWindows()


# angle = np.arctan2(1.732, 3)
# print(angle/np.pi*180)

# print(np.uint8(257))
import matplotlib.pyplot as plt
import numpy as np
import cv2

# x = np.array([3, 4, 6])
# x1 = [i*20 for i in range(3)]
# plt.bar(x1, x, 20, align='edge')
# # plt.xticks(range(0, 40, 20))
# plt.show()

# print(x1)


# class boy:
#     def __init__(self):
#         super(boy, self).__init__()
#         self.x = self.Add(5)
#     def Add(self, y):
#         return y
#     def resize_img(self, img):
#         return cv2.resize(img, (100, 200))
#     def forward(self):
#         # img_re = self.resize_img(img)(img)
#         x0 = self.x
#         # x1 = self.x(x0)
#         return x0
# if __name__ == '__main__':
#     img = cv2.imread('photo/canon.jpg')
#     boy1 = boy()
#     # img_re = boy1.forward(img)
#     # print(img_re.shape)
#     x15 = boy1.forward()
#     print(x15)


# -*- coding: utf-8 -*-
# By：Eastmount
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
im = cv2.imread('photo/wyz.jpg', 0)

# 设置鼠标左键开启
en = False


# 鼠标事件
def draw(event, x, y, flags, param):
    global en
    # 鼠标左键按下开启en值
    if event == cv2.EVENT_LBUTTONDOWN:
        en = True
    # 鼠标左键按下并且移动
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_LBUTTONDOWN:
        # 调用函数打马赛克
        if en:
            drawMask(y, x)
        # 鼠标左键弹起结束操作
        elif event == cv2.EVENT_LBUTTONUP:
            en = False


# 图像局部采样操作
def drawMask(x, y, size=10):
    # size*size采样处理
    m = int(x / size * size)
    n = int(y / size * size)
    print(m, n)
    # 10*10区域设置为同一像素值
    for i in range(size):
        for j in range(size):
            im[m + i][n + j] = im[m][n]


# 打开对话框
cv2.namedWindow('image')

# 调用draw函数设置鼠标操作
cv2.setMouseCallback('image', draw)

# 循环处理
while (1):
    cv2.imshow('image', im)
    # 按ESC键退出
    if cv2.waitKey(10) & 0xFF == 27:
        break
    # 按s键保存图片
    elif cv2.waitKey(10) & 0xFF == 115:
        cv2.imwrite('sava.png', im)

# 退出窗口
cv2.destroyAllWindows()



