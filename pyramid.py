import cv2


# 定义高斯金字塔
def pyramid(img):
    level = 3
    temp = img.copy()
    pyramid_img = []
    for i in range(level):
        res = cv2.pyrDown(temp)
        pyramid_img.append(res)
        cv2.imshow('pyramid'+str(i), res)
        temp = res.copy()
    return pyramid_img      # 返回值为存有三次向下采样后的图像列表


# 拉普拉斯金字塔  先将低分辨率图像向上采样，再用原图减去采样图像，得到拉普拉斯图像
def lapalian_img(img):
    pyramid_img = pyramid(img)
    level = len(pyramid_img)    # pyramid_img列表内有3个图像
    for i in range(level, 0, -1):     # 3, 2, 1
        if i == 1:
            # res = cv2.pyrUp(pyramid_img[i - 1], dstsize=img.shape[:2])   # 对图像向上采样,保证图像尺寸大小与原图想等
            res = cv2.pyrUp(pyramid_img[i - 1])
            lpls = cv2.subtract(img, res)    # 最后一层用原图减去采样图像
            cv2.imshow('lapalian_img' + str(i), lpls)
        else:
            # res = cv2.pyrUp(pyramid_img[i - 1], dstsize=pyramid_img[i - 2].shape[:2])  # 保证图像尺寸大小与高斯金字塔对应层想等
            res = cv2.pyrUp(pyramid_img[i - 1])
            lpls = cv2.subtract(pyramid_img[i - 2], res)
            cv2.imshow('lapalian_img' + str(i), lpls)


if __name__ == '__main__':
    img = cv2.imread('canon.jpg')
    img = cv2.resize(img, (512, 256))   # 宽512 高256
    cv2.imshow('input_img', img)
    pyramid(img)
    lapalian_img(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()