from PIL import Image, ImageDraw
import cv2
import numpy as np


def clamp(a):
    if a>255:
        return 255
    elif a<0:
        return 0
    else:
        return a


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


def free_noise():
    img = Image.new('L', (300, 400), 'black')
    noise = ImageDraw.Draw(img)
    noise.rectangle((10, 10, 40, 40), fill='white')
    noise.arc((60, 60, 150, 150), 0, 360, fill='white', width=10)
    noise.rectangle((20, 20, 25, 25), fill='black')
    noise.rectangle((200, 200, 255, 225), fill='white')
    noise.arc((250, 250, 260, 260), 0, 270, fill='white')

    img.show()
    img.save('E:\\pyitem\\opencv_img\\photo\\xingzhuang1.jpg')




if __name__ == '__main__':
    # img = Image.new('RGB', (320, 270), 'white')
    # img.save('empty.jpg')
    # img1 = cv2.imread('empty.jpg')
    img1 = cv2.imread('E:\\pyitem\\opencv_img\\photo\\xingzhuang.jpg')
    free_noise()
    print(img1.shape)
    # gaussian_noise(img1)
    # cv2.imwrite('E:\\pyitem\\opencv_img\\photo\\wzm_noise.jpg', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



