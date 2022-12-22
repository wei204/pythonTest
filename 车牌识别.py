import cv2
import imutils
import numpy as np
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

img = cv2.imread('./photo/car.jpg',cv2.IMREAD_COLOR)
img = cv2.resize(img, (600,400) )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 13, 15, 15)
# cv2.imshow('filter_img', gray)

edged = cv2.Canny(gray, 30, 200)
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for i in range(0, len(contours)):
#     length = cv2.arcLength(contours[i], True)
#     if length > 20:
#         cv2.drawContours(edged, contours[i], -1, (0, 0, 255), 2)
cv2.imshow('boundary', edged)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:20]
screenCnt = None

for c in contours:
    # 计算轮廓周长
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, False)
    cv2.drawContours(img, [approx], -1, [0, 0, 255])
    cv2.imshow('img', img)
    # if peri>200 and peri<300:
    #     screenCnt = approx
    # if len(approx) == 4:
    #     screenCnt = approx
    # break
# print(screenCnt)
# if screenCnt is None:
#     detected = 0
#     print ("No contour detected")
# else:
#      detected = 1
#
# if detected == 1:
#     cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

# mask = np.zeros(gray.shape,np.uint8)
# new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
# cv2.imshow('new', new_image)
#
# (x, y) = np.where(mask == 255)
# (topx, topy) = (np.min(x), np.min(y))
# (bottomx, bottomy) = (np.max(x), np.max(y))
# Cropped = gray[topx:bottomx+1, topy:bottomy+1]
#
# text = pytesseract.image_to_string(Cropped, config='--psm 11')
# print("programming_fever's License Plate Recognition\n")
# print("Detected license plate Number is:",text)
# img = cv2.resize(img,(500,300))
# Cropped = cv2.resize(Cropped,(400,200))
# cv2.imshow('car',img)
# cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()