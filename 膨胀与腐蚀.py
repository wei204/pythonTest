import cv2


# 腐蚀
def erode_img(img):
    # 将图像转为灰度图并进行二值化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
    cv2.imshow('binary', binary)
    # 定义腐蚀内核形状与大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    res = cv2.erode(img, kernel)
    cv2.imshow('erode_img', res)


# 膨胀
def dilate_img(img):
    # 将图像转为灰度图并进行二值化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
    cv2.imshow("binary", binary)
    # 定义腐蚀内核形状与大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    res = cv2.dilate(binary, kernel)
    cv2.imshow('dilate_img', res)


# 开操作 先腐蚀再膨胀 开运算可以用来消除小黑点 图像整体粗细几乎不变，光滑断开
def open_img(img):
    # 将图像转为灰度图并进行二值化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
    cv2.imshow("binary", binary)
    # 定义腐蚀内核形状与大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    res = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imshow('open_img', res)


# 闭操作  图像整体粗细几乎不变，光滑连接
def close_img(img):
    # 将图像转为灰度图并进行二值化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
    cv2.imshow("binary", binary)
    # 定义腐蚀内核形状与大小
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    res = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('close_img', res)


if __name__ == '__main__':
    img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\xingzhuang1.jpg')
    cv2.imshow('original', img)
    # print(img.shape)
    # erode_img(img)
    dilate_img(img)
    # capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # while True:
    #     ret, frame = capture.read()
    #     dilate_img(frame)
    #     if cv2.waitKey(10) & 0xff == 27:
    #         break
    # capture.release()
    # open_img(img)
    # close_img(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()