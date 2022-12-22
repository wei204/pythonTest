import cv2
import numpy as np

# img1 = cv2.imread('me.jpg')
# face = img1[100:400, 100:300]      # 高度，宽度
# # 转为灰度图
# gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
# # print(gray.shape)    # (300, 200) 处理后为一维数组，需要将其转换为三维数组，才能与bgr图像融合
# # 用灰度图替换原图中face
# face_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   # 转为三通道，但是仍然为灰度图
# img1[100:400, 100:300] = face_bgr
# print(img1.shape)          # (559, 398, 3)  opencv中先显示高度后显示宽度
# # cv2.imshow('face', gray)
# cv2.imshow('original', img1)
# cv2.waitKey()
# cv2.destroyAllWindows()

# 泛洪填充

# 定义填充函数
def fill_color(img):
    # copy_img = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)   # 高度宽度加2是为了处理边界时更方便，mask类型必须为uint8格式,区域内为0的被泛洪
    cv2.floodFill(img, mask, (10, 10), (255, 255, 255), (100, 100, 100), (120, 120, 120), cv2.FLOODFILL_FIXED_RANGE)
                  # 从(50,50)处开始对灰度值小于该点灰度值90，大于该点20灰度值的区域填充为黄色
    # 平滑处理
    img_smooth = cv2.pyrMeanShiftFiltering(img, 10, 20)
    cv2.imshow('fill', img_smooth)


if __name__ == '__main__':
    # img2 = cv2.imread('E:\\pyitem\\opencv_img\\photo\\me.jpg')
    # fill_color(img2)
    # cv2.waitKey(0)

    # 调用摄像头
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = capture.read()
        fill_color(frame)
        if cv2.waitKey(10) & 0xff == 27:
            break
    capture.release()

    cv2.destroyAllWindows()


# 二值化填充
# def fill_binary():
#     img = np.zeros([200, 200, 3], np.uint8)
#     img[50:150, 50:150, :] = 255
#     mask = np.ones([202, 202], np.uint8)
#     mask[50:150, 50:150] = 0
#     cv2.floodFill(img, mask, (50, 50), (0, 255, 255), cv2.FLOODFILL_MASK_ONLY)
#     cv2.imshow('fill', img)
#
# if __name__ == '__main__':
#
#     fill_binary()
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
