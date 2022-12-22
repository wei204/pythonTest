import cv2
import numpy as np


def nothing(x):
    pass

me = cv2.imread(r'E:\pyitem\opencv_img\photo\me.jpg')
wyz = cv2.imread(r'E:\pyitem\opencv_img\photo\wyz.jpg')
wyz = cv2.resize(wyz, (me.shape[1], me.shape[0]))

window = np.zeros([me.shape[0], me.shape[1], 3], np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('value', 'image', 0, 100, nothing)


while(1):
    cv2.imshow('image', window)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    r = cv2.getTrackbarPos('value', 'image')
    r = float(r) / 100.0
    window=cv2.addWeighted(wyz, r, me, 1.0-r, 0)

cv2.destroyAllWindows()
#




#############################只换脸###############################################
# import cv2
# import numpy as np

# def nothing(x):
#     pass

# me = cv2.imread(r'E:\pyitem\opencv_img\photo\me.jpg')
# wyz = cv2.imread(r'E:\pyitem\opencv_img\photo\wyz.jpg')
#
# me_face = me[100:400, 100:300]
# wyz_face = wyz[80:300, 70:230]
# wyz_face = cv2.resize(wyz_face, (me_face.shape[1], me_face.shape[0]))
# # cv2.imshow('me_face', me_face)
# # cv2.imshow('wyz_face', wyz_face)
#
# image = np.zeros([me.shape[0], me.shape[1], 3], np.uint8)
# cv2.namedWindow('image')
#
# cv2.createTrackbar('value', 'image', 0, 100, nothing)
# while 1:
#     cv2.imshow('image', image)
#     k = cv2.waitKey(1) & 0xff
#     if k == 27:
#         break
#     r = cv2.getTrackbarPos('value', 'image')
#     r = float(r)/100.0
#     image = cv2.addWeighted(wyz_face, r, me_face, 1-r, 0)
#
# cv2.destroyAllWindows()
