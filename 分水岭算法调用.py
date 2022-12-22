import cv2

img = cv2.imread('./photo/xingzhuang.jpg')
# cv2.imshow('img', img)

#灰度化二值化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow('binary', binary)
img_contours, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(hierarchy.shape)
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cv2.imshow('img', img)

# retval, markers = cv2.connectedComponents(img)
# cv2.watershed(img, markers)
# # print(markers[])
# cv2.imshow("1",markers)

cv2.waitKey(0)
cv2.destroyAllWindows()