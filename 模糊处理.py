import cv2
import numpy as np

# 模糊处理 均值滤波  处理随机噪声比较好
# res = cv2.blur(img1, (3, 3))     # 3*3区域
# 中值滤波 处理椒盐噪声
# res = cv2.medianBlur(img1, 3)     # 方框尺寸为3*3


# 自定义模糊或锐化
def free_blur(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) # 算子之和为1一般为锐化，为0一般为边缘锐化
    res = cv2.filter2D(img, -1, kernel=kernel)   # -1 为默认
    cv2.imshow('free_blur', res)

# 大于255时取255，小于0时取0
def clamp(a):
    if a > 255:
        return 255
    elif a < 0:
        return 0
    else:
        return a

# 产生高斯噪声
def gaussian_noise(img):
    h, w, ch = img.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)   # 产生正态分布 0代表以y轴对称，20是标准差
            # img[row, col, 0] = clamp(img[row, col, 0] + s[0])
            # img[row, col, 1] = clamp(img[row, col, 1] + s[1])
            # img[row, col, 2] = clamp(img[row, col, 2] + s[2])      # 加入噪声
            b = img[row, col, 0]
            g = img[row, col, 1]
            r = img[row, col, 2]
            img[row, col, 0] = clamp(b+s[0])
            img[row, col, 1] = clamp(g + s[1])
            img[row, col, 2] = clamp(r + s[2])
    cv2.imshow('gaussian_noise', img)

# 双边滤波
def bi_filter(img):
    res = cv2.bilateralFilter(img, 0, 100, 15)     # 第二个参数为非正数时，其大小由第四个参数自动计算。
                                                  # 第三个参数为灰度滤波器的sigma值；第四个参数为空间滤波器的sigma值,不宜选过大。
    # return res
    cv2.imshow('bi_filter', res)

# 均值漂移滤波
def shift(img):
    res = cv2.pyrMeanShiftFiltering(img, 10, 25)  # 第二个参数为空间半径，第三个参数为色彩半径
    # return res
    cv2.imshow('shift', res)
# 像素减法
def img_sub(img1, img2):
    img1 = cv2.resize(img1, (398, 559))
    img2 = cv2.resize(img2, (398, 559))
    res = cv2.subtract(img1, img2)
    return res
if __name__ == '__main__':
    # img1 = cv2.imread('C:/Users/Administrator/Pictures/Camera Roll/wei.jpg')
    # img1 = cv2.resize(img1, (300, 400))
    # free_blur(img1)
    # cv2.imshow('original', img1)
    # t1 = cv2.getTickCount()
    # gaussian_noise(img1)
    # t2 = cv2.getTickCount()
    # # 运行时间
    # t = (t2-t1)/cv2.getTickFrequency()
    # print('时间消耗：', t)
    # 高斯模糊
    # res = cv2.GaussianBlur(img1, (5, 5), 0)
    # 中值滤波
    # res = cv2.medianBlur(img1, 3)
    # cv2.imshow('blur', res)
    # bi_filter(img1)
    # shift(img1)
    # free_blur(res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 创建logo
    img1 = cv2.imread('wzm.jpg')
    img2 = cv2.imread('me1.jpg')
    img3 = cv2.imread('star.jpg')
    img2 = cv2.resize(img2, (398, 559))
    # print(img1.shape, img2.shape)

    # cv2.imshow('res', res)
    # word = img1[200:400, 0:398]
    word = img_sub(img1, img3)
    res = cv2.add(word, img2)
    cv2.imshow('word', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

