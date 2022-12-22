import cv2
import numpy as np

def gray_color(img):
    zd = np.zeros((400, 720, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            # gray[i, j] = gray[i, j] / 255
            # b = np.uint8(np.round(gray[i, j] * np.random.uniform(1.031, 1.080)))
            # g = np.uint8(np.round(gray[i, j] * np.random.uniform(1.007, 1.015)))
            # r = np.uint8(np.round(gray[i, j] * np.random.uniform(0.94, 0.96)))
            b = np.uint8(np.round(gray[i, j] * np.random.uniform(1, 1.5)))
            g = np.uint8(np.round(gray[i, j] * np.random.uniform(1, 1.1)))
            r = np.uint8(np.round(gray[i, j] * np.random.uniform(0.3, 1)))
            # print(b)
            # print(g)
            # print(r)
            zd[i, j, 0] = b
            zd[i, j, 1] = g
            zd[i, j, 2] = r
    return zd

def sat_num(data):
    if data > 255:
        data = 255
    elif data < 0:
        data = 0
    else:
        data = data
    return data

def img_Gamma(img, gamma):
    # 建立对应表
    LTU = np.zeros(256)
    for i in range(256):
        f = (i + 0.5) / 255
        f = np.power(f, 1/gamma)
        LTU[i] = np.round(sat_num(f*255-0.5))
        # LTU[i] = sat_num(f * 255 - 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            gray[i, j] = LTU[gray[i, j]]
    return gray
    # return LTU





if __name__ == '__main__':

    img1 = cv2.imread('./photo/hei.jpg')
    # cv2.imshow('original', img1)
    # gray_color(img1)
    # zd = np.zeros((400, 720, 3))
    # print(img1.shape)
    # print(zd.shape)

    # zd = gray_color(img1)
    # cv2.imshow('zd', zd)
    # print(img1)



    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img[100, 13])
    # print(gray[100, 13])

    # a = np.zeros((100,200,3))
    # for i in range(100):
    #     for j in range(200):
    #         a[i, j, 0] = np.random.randint(0, 255, 1)
    #         a[i, j, 1] = np.random.randint(0, 255, 1)
    #         a[i, j, 2] = np.random.randint(0, 255, 1)
    # # print(a)
    # cv2.imshow('a', a)
    # print(a)

    # Gamma变换
    gray_Gamma = img_Gamma(img1, 0.5)
    cv2.imshow('Gamma', gray_Gamma)
    # print(gray_Gamma)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # 视频处理
    # capture = cv2.VideoCapture('E:\\pyitem\\opencv_img\\photo\\zhude.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps = capture.get(cv2.CAP_PROP_FPS)
    # size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # out = cv2.VideoWriter('./photo/zhude1.avi', fourcc, fps, size)
    # while(True):
    #     ret, frame = capture.read()
    #     if ret==True:
    #         frame_Gamma = img_Gamma(frame, 0.5)
    #         cv2.imshow('Gamma', frame_Gamma)
    #
    #
    #         # 向视频文件写入一帧
    #         out.write(frame_Gamma)
    #     if cv2.waitKey(100) & 0xff == 27:
    #         break
    # capture.release()
    # cv2.destroyAllWindows()
