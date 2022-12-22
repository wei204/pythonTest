# import cv2
# import numpy as np
#
# img = cv2.imread(r'E:\pyitem\opencv_img\photo\me.jpg')
# # print(img.shape)
# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# blue_low = np.array([100, 43, 46])
# blue_hight = np.array([124, 255, 255])
# mask = cv2.inRange(img_hsv, blue_low, blue_hight)
# # cv2.imshow('mask', mask)
# # red_low = np.array([156, 43, 46])
# # red_hight = np.array([180, 255, 255])
# red = cv2.imread(r'E:\pyitem\opencv_img\photo\red.jpg')
# red = cv2.resize(red, (398, 559))
#
# # print(mask_red.shape)
# # mask_red_b = cv2.bitwise_and(mask, mask, mask=mask_red)
#
# mask_not = cv2.bitwise_not(mask)
# # cv2.imshow('mask_not', mask_not)
# res = cv2.bitwise_and(img, img, mask=mask_not)
#
# res_red = res + red
# # res_red = cv2.bitwise_and(res, res, mask=red)
# # cv2.imshow('oright', img)
# cv2.imshow('res', res)
# cv2.imshow('res_red', res_red)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

