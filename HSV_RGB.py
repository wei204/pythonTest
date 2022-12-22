import matplotlib.pyplot as plt
import cv2


# 利用plt提取bgr
# img = plt.imread(r'./photo/canon.jpg')
# # plt.imshow(img[:, :, 0])    # 红色通道
# # plt.show()
# print(img[:, :, 0].shape)


# 利用opencv提取bgr
# img = cv2.imread(r'./photo/canon.jpg')
# cv2.imshow('blue', img[:, :, 0])
# cv2.imshow('green', img[:, :, 1])
# cv2.imshow('red', img[:, :, 2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # opencv自带函数提取bgr
# img = cv2.imread(r'photo/canon.jpg')
# b, g, r = cv2.split(img)
# cv2.imshow("Blue 1", b)
# cv2.imshow("Green 1", g)
# cv2.imshow("Red 1", r)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# 将rgb通道转为hsv

def rgb_hsv(img):

    for height in range(img.shape[0]):
        for width in range(img.shape[1]):
            # print(img[h, w])
            r, g, b = img[height, width][0], img[height, width][1], img[height, width][2]
            # 归一化
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0
            max_color = max(r, g, b)
            min_color = min(r, g, b)
            v = max_color
            delta = (max_color - min_color)

            if (v != 0):
                s = delta/max_color
            else:
                s = 0


            # if v > 0:
            #     s = (v - min_color) / max_color
            # else:
            #     s = 0

            if(delta!=0):
                if v == r:
                    h = 60 * (g - b) / delta
                elif v == g:
                    h = 60 * (b - r) / delta
                else:
                    h = 60 * (r - g) / delta


            else:
                h = 0
            # if h < 0:
            #     h = h + 360
            # else:
            #     h = h
            img[height, width][0], img[height, width][1], img[height, width][2] = h, s, v

    return img


if __name__ == '__main__':
    img = plt.imread(r'./photo/me.jpg')
    # print(img.shape[0])
    # img1 = cv2.imread("./photo/canon.jpg")
    # img2 = cv2.imread('./photo/canon_hsv.jpg')
    # print(img)
    # print(img2)
    img_hsv = rgb_hsv(img)
    cv2.imshow('hsv', img_hsv)
    img1 = cv2.add(img, img_hsv)
    # plt.imshow(img1)
    # plt.imshow(img_hsv)

    cv2.imshow('img1', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img = cv2.imread('./photo/canon.jpg')
    # img_hsv_opencv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv", img_hsv_opencv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # plt.imshow(img_hsv_opencv)
    # plt.show()


