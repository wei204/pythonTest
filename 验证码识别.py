import cv2
import pytesseract as tess
from PIL import Image


def recognize_text(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('binary', binary)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # dilate = cv2.dilate(binary, kernel)
    # cv2.imshow('dilate', dilate)
    # cv2.bitwise_not(dilate, dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imshow('binary_open', binary_open)
    # # textImage = Image.fromarray(dilate_open)
    # binary_open = cv2.bitwise_not(binary_open, binary_open)
    # cv2.imshow('binary1', binary_open)
    textImage = Image.fromarray(binary_open)
    text = tess.image_to_string(textImage)
    print(text)


if __name__ == '__main__':
    img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\1234.jpg')
    recognize_text(img)
    print(tess.get_languages())
    cv2.waitKey(0)
    cv2.destroyAllWindows()