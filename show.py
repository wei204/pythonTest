import cv2
import matplotlib.pyplot as plt
# 显示函数
def img_show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_RGB)
    plt.show()