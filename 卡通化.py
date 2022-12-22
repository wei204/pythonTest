import cv2
import numpy as np


# 强化边缘
def edge_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    # cv2.imshow('gray_blur', gray_blur)
    # 强化边缘
    edge = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    # cv2.imshow('edge', edge)
    return edge

# 颜色量化
def color_quality(img, k):
    data = np.float32(img).reshape((-1, 3))
    # （type,max_iter,epsilon）
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 0.001)  #终止准则为最大迭代次数20+精度0.001
    flag = cv2.KMEANS_RANDOM_CENTERS
    ret, bestLabels, centers = cv2.kmeans(data, k, None, criteria, 10, flag)  #10代表随机初始化10次寻找初始质心
    # 将数组转为二维图像
    centers = np.uint8(centers)
    result = centers[bestLabels.flatten()]   #将簇中所有像素用质心表示
    result = result.reshape(img.shape)
    # cv2.imshow('color_quality', result)
    return result


if __name__ == '__main__':
    # 图像卡通化
    img = cv2.imread(r'E:\pyitem\opencv_img\photo\me.jpg')
    mask = edge_mask(img)
    img_color = color_quality(img, 3)
    cv2.imshow('img_color', img_color)
    # 结合边缘与颜色模糊后的图片
    img_res = cv2.bitwise_and(img_color, img_color, mask=mask)
    cv2.imshow('result', img_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # # 视频卡通化
    # # capture = cv2.VideoCapture(r'E:\pyitem\opencv_img\photo\nba.mp4')
    # capture = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = capture.read()
    #     if ret is True:
    #         # frame = frame[480:900, 0:720]
    #         mask = edge_mask(frame)
    #         frame_color = color_quality(frame, 3)
    #         frame_res = cv2.bitwise_and(frame_color, frame_color, mask=mask)
    #         cv2.imshow('res', frame_res)
    #
    #     if cv2.waitKey(1) & 0xff == 27:
    #         break
    # capture.release()
    # cv2.destroyAllWindows()
