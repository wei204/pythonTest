import cv2


def contours_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 返回值为阈值和二值图像
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv2.drawContours(img, contours, i, (0, 0, 255), 2)
    cv2.imshow('contours_img', img)


if __name__ == '__main__':
    img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\xingzhuang.jpg')
    # capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # while True:
    #     ret, frame = capture.read()
    #     contours_img(frame)
    #     if cv2.waitKey(50) & 0xFF == 27:
    #         break
    # capture.release()
    contours_img(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()