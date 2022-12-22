import cv2
import matplotlib .pyplot as plt
import numpy as np
from PIL import Image

# 读入图片
# img1 = cv2.imread('canon.jpg')
# 显示BGR
# print(img1)
# # 显示图片
# cv2.imshow('qiao',img1)
# cv2.waitKey(0)            # 等待按键触发函数，参数为0时窗口一直显示  单位为ms
# cv2.destroyAllWindows()     # 销毁窗口函数

# 显示函数
def cv2_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读入灰度图
# img2 = cv2.imread('canon.jpg',cv2.IMREAD_GRAYSCALE)
# # cv2_show('hui',img2)
# # print(img2)
# # 图像保存
# # cv2.imwrite('hcanon.jpg',img2)
# print(type(img2))                  # <class 'numpy.ndarray'>
# print(img2.size)
# print(img2.dtype)            # uint8

# 将灰度图转为彩色图
# img3 = cv2.imread('hcanon.jpg')
# img4 = cv2.cvtColor(img3,cv2.COLOR_GRAY2BGR)
# cv2_show('cai',img4)
# # 将彩色图转为灰度图
# 错误
# im = Image.open("canon.jpg")
# pix = im.load()
# width = im.size[0]
# height = im.size[1]
# c = []
# for x in range(width):
#     for y in range(height):
#         r, g, b = pix[x, y]
#         c1 = r * 0.3 + g * 0.6 + b * 0.1
#         c.append(c1)
# print(c)


# img5 = cv2.imread('canon.jpg')
# img6 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
# # cv2_show('cai',img6)
# print(img6)           # 图像灰度值

# # 分解bgr分量
# img7 = cv2.imread('canon.jpg')
# b,g,r = cv2.split(img7)
# print(img7.shape)
# # 合并三通道分量
# img8 = cv2.merge((b,g,r))
# print(img8.shape)
# # 只保留一个通道
# img9 = img8.copy()
# img9[:,:,0] = 0
# img9[:,:,1] = 0
# cv2_show('R',img9)

# # 边界填充
# img0 = cv2.imread('daice.jpg')
# # 指定边界填充尺寸
# top,bottoms,left,right = (50,50,50,50)
# # 填充方式
# replicate = cv2.copyMakeBorder(img0,top,bottoms,left,right,borderType=cv2.BORDER_REPLICATE)    # 最边界法
# reflect = cv2.copyMakeBorder(img0,top,bottoms,left,right,borderType=cv2.BORDER_REFLECT)    # 反射法 以边界为轴
# reflect101 = cv2.copyMakeBorder(img0,top,bottoms,left,right,borderType=cv2.BORDER_REFLECT_101)    # 反射法
# wrap = cv2.copyMakeBorder(img0,top,bottoms,left,right,borderType=cv2.BORDER_WRAP)    # 外包装法
# constant = cv2.copyMakeBorder(img0,top,bottoms,left,right,borderType=cv2.BORDER_CONSTANT,value=200)    # 常数法
# # 窗口显示
# plt.subplot(111),plt.imshow(img0),plt.title('original')
# # plt.subplot(212),plt.imshow(replicate,'gray'),plt.title('replicate')
# # plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('reflect')
# # plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('reflect101')
# # plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('wrap')
# # plt.subplot(236),plt.imshow(constant,'gray'),plt.title('constant')
# plt.show()

# # 基本数值计算
# img10 = cv2.imread('canon.jpg')
# # img10 = img10 + 10
# # cv2_show('10',img10)
# print(img10.shape)        # (266, 400, 3)
# print(img10.size)        # 266*400*3=319200
# s = img10.shape
# w,h,n = s[0],s[1],s[2]
# print('图像的高度、宽度、维数分别为',w,h,n)

# # 图像融合
# img11 = cv2.imread('canon.jpg')
# img12 = cv2.imread('daice.jpg')
# # 调整图像尺寸
# img13 = cv2.resize(img12,(400,266))
# # 融合
# res = cv2.addWeighted(img11,0.6,img13,0.4,10)     # 像素值res=img11*0.6+img13*0.4+10
# cv2_show('融合',res)

# 按比例改变图像像素
# img14 = cv2.imread('canon.jpg')
# res = cv2.resize(img14,(0,0),fx=2,fy=2)
# print(res.shape)     # (532, 800, 3)
# # cv2_show('change',res)


# 显示函数
def img_show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_RGB)
    plt.show()


# 显示图像
img0 = cv2.imread(r'E:\pyitem\opencv_img\photo\canon.jpg')
# print(img0)
img_show(img0)
# opencv按照BGR存储图像，matplotlib按照RGB存储图像
# img0_RGB = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
# plt.imshow(img0_RGB)
# plt.show()
img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img_show(img0_gray)
# plt.imshow(img0_gray, cmap='gray')
# plt.show()


