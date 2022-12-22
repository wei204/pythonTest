import cv2
import numpy as np


def bgr_hsv():
    # 读入视频文件
    video = cv2.VideoCapture('C:\\Users\\Administrator\\Pictures\\Camera Roll\\diver.mp4')
    while True:
        ret, frame = video.read()
        if ret == False:
            break
        # 将每帧图像转为hsv格式
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 转为二值图像
        lower_hsv = np.array([0, 0, 0])
        upper_hsv = np.array([180, 255, 46])
        res = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        # 像素运算只显示黑
        res1 = cv2.bitwise_and(frame, frame, mask=res)

        # 显示处理后的视频
        cv2.imshow('res', res1)

        c = cv2.waitKey(10)
        if c == 27:
            break


if __name__ == '__main__':
    bgr_hsv()
    cv2.waitKey(0)
    cv2.destroyAllWindows()



