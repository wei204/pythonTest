import numpy as np
import cv2
import matplotlib.pyplot as plt


def img_grad(img):
    h, w = img.shape
    img_x = np.zeros((h, w), np.uint8)
    img_y = np.zeros((h, w), np.uint8)
    img_xy = np.zeros((h, w), np.uint8)
    img_angle = np.zeros((h, w), np.uint8)
    img_pad = cv2.copyMakeBorder(img, 1,1,1,1,cv2.BORDER_REPLICATE)
    #计算x方向、y方向的梯度
    for i in range(1, h+1):
        for j in range(1, w+1):
            img_x[i-1, j-1] = 0.5*(img_pad[i, j+1] - img_pad[i, j-1])   #x方向梯度
            img_y[i-1, j-1] = 0.5*(img_pad[i+1, j] - img_pad[i-1, j])  #y方向梯度 显示水平边缘
    #计算总梯度和梯度方向
    for i in range(h):
        for j in range(w):
            img_xy[i, j] = np.sqrt(img_x[i, j]**2+img_y[i, j]**2)
            img_angle[i, j] = np.arctan2(img_y[i, j], img_x[i, j]) * 180 / np.pi
    return img_xy, img_angle


#获取每个cell（8*8）区域内梯度方向的直方图
def HOG_feature(img):
    img_xy, img_angle = img_grad(img)
    grad_hist = np.zeros((128, 9))

    # print(grad_hist.shape)
    # print(grad_hist[2])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img_angle[i, j]<20):
                grad_hist[0] += (20-img_angle[i, j])/20*img_xy[i, j]
                grad_hist[1] += img_angle[i, j]/20*img_xy[i, j]
            elif(img_angle[i, j]<40):
                grad_hist[1] += (40-img_angle[i, j])/20*img_xy[i, j]
                grad_hist[2] += (img_angle[i, j]-20)/20*img_xy[i, j]
            elif (img_angle[i, j] < 60):
                grad_hist[2] += (60 - img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[3] += (img_angle[i, j]-40) / 20 * img_xy[i, j]
            elif (img_angle[i, j] < 80):
                grad_hist[3] += (80 - img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[4] += (img_angle[i, j]-60) / 20 * img_xy[i, j]
            elif (img_angle[i, j] < 100):
                grad_hist[4] += (100 - img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[5] += (img_angle[i, j]-80) / 20 * img_xy[i, j]
            elif (img_angle[i, j] < 120):
                grad_hist[5] += (120 - img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[6] += (img_angle[i, j]-100) / 20 * img_xy[i, j]
            elif (img_angle[i, j] < 140):
                grad_hist[6] += (140 - img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[7] += (img_angle[i, j]-120) / 20 * img_xy[i, j]
            elif (img_angle[i, j] < 160):
                grad_hist[7] += (160 - img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[8] += (img_angle[i, j]-140) / 20 * img_xy[i, j]
            else:
                grad_hist[8] += (180-img_angle[i, j]) / 20 * img_xy[i, j]
                grad_hist[0] += (img_angle[i, j] - 160)/20*img_xy[i, j]
    return grad_hist

if __name__ == '__main__':
    img = cv2.imread('photo/people.jpg')
    img = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #绘制梯度方向分布直方图
    grad_hist = HOG_feature(gray)
    x = [i*20 for i in range(9)]
    plt.bar(x, grad_hist, 20, align='edge')
    plt.xticks([i*20 for i in range(9)])
    plt.show()

    # cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()