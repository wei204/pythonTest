import cv2
import time

# vc = cv2.VideoCapture('gaotie.mp4')
# # 检查是否读取正确即是否打开
# if vc.isOpened():
#     oepn,frame = vc.read()    # ret,frame是获cap.read()方法的两个返回值。
# else:                         # 其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就为False。
#      oepn = False                          # frame就是每一帧的图像，是个三维矩阵。
# while oepn:
#     ret,frame = vc.read()
#     if frame is None:
#         break
#     if ret == True:
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)          # 将每一帧图片改为灰度图
#         cv2.imshow('result',gray)
#         if cv2.waitKey(10) & 0xFF == 27:                         # ASCII码为27对应退出键ESC
#             break
# vc.release()
# cv2.destroyAllWindows()


# # 读入高铁测试视频
capture = cv2.VideoCapture('E:\\pyitem\\opencv_img\\photo\\nba.mp4')
# capture = cv2.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if ret is True:
        frame = frame[480:900, 0:720]
        # print(frame.shape)
        cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xff == 27:
        break
capture.release()
cv2.destroyAllWindows()

