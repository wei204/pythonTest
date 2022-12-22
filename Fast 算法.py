import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("E:\\pyitem\\opencv_img\\photo\\circle.jpg")
# cv2.imshow('canon', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img2)a

fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img)
img_out = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
cv2.imshow('img', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()