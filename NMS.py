import numpy as np
import cv2


# def my_nms(dets, thresh):
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     score = dets[:, 4]
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     order = np.argsort(score)[::-1]

# img = cv2.imread('./photo/xingzhuang.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(img, None)
# cv2.drawKeypoints(gray, kp, img)
# cv2.imshow('keypoint', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# data = []
# with open('./data/data.text', 'r') as f:
    # 一行行读取
    # line = f.readline().strip('\n')
    # # line = f.readline().split('\n')[0]  #效果一致
    # data.append(line)

    # # 全部读取
    # lines = f.readlines()
    # for line in lines:
    #     line = line.strip('\n')
    #     data.append(line)

# print(data)
# index = len(data)
# print(index)
# with open('./data/val.text', 'w') as f1:
#     for i in range(3):
#         # if i!=0: f1.write('\n'):
#         pass

#
# for i in range(5):
#     if (i!=0:3):
#         print('45')




# import xml.etree.ElementTree as ET
# # tree = ET.parse('E:/pyitem/steel/insect_det/Annotations/0001.xml')
# tree = ET.parse('./data/country.xml')
# root = tree.getroot()
#
# # print(root.tag)
# # for child in root:
#     # print(child.tag, child.attrib)
#     # print(root[0][1].text)
#
# print(root[0][1])


img = cv2.imread('./photo/canon.jpg')
print(img.shape)
print(img.shape[-2:])
