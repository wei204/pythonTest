# #开始导入需要的模块
# import cv2#调用摄像头
# import dlib#调用识别检测库
# from math import hypot
# import time
# import winsound#调用设备（笔记本）音响，提示疲劳
#
# cap = cv2.VideoCapture(0)#打开笔记本的内置摄像头，参数改为视频位置则为打开视频文件
# detector = dlib.get_frontal_face_detector()#获取人脸分类器
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")#获取人脸检测器，提取特征点数
# font = cv2.FONT_HERSHEY_PLAIN#设置写入文字的字体（在屏幕上的字体）
#
# #用于求上眼皮与下眼皮的重点
# def midpoint(p1 ,p2):
#     return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)
#
# #用于计算眼睛长宽比，获取比值
# def get_blinking_ratio(eye_points, facial_landmarks):
#     left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
#     right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
#     #利用脸谱特征图上的点，获得人脸上眼睛两边的坐标
#
#     center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
#     center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
#     #利用脸谱特征图上的点，获得人脸上眼睛上下眼皮的坐标，同时计算中间点的坐标
#
#     hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 3)
#     ver_line = cv2.line(frame, center_top, center_bottom, (0,255,255), 3)
#     #将眼睛左右与上下连成线，方便观测
#
#     hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
#     ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
#     #利用hypot函数计算得出线段的长度
#
#     ratio = hor_line_lenght / ver_line_lenght
#     #得到长宽比
#     return ratio
#
# #主程序，一直检测眼睛睁眨，长宽比与一个定值（临界点）比较，判断是否疲劳且发出提示音
# while True:
#     _, frame = cap.read()#这个read是cv2中的方法，作用：按帧读取画面，返回两个值（True，frame）有画面是True，且赋值给frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#方法，作用：将摄像头捕获的视频转换为灰色并且保存，这样方便判断面部特征点
#     faces = detector(gray)#利用dlib库，处理获取的人脸画面
#
#     #循环每一个画面
#     for face in faces:
#         landmarks = predictor(gray, face)
#         left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
#         right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
#         #利用函数获得左右眼的比值
#
#         blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
#         #取平均数
#
#         end = time.time()#记时，判断闭眼时间
#
#         #检测眼睛状况
#         if blinking_ratio > 4.5:
#             cv2.putText(frame, "CLOSE", (75, 250), font, 7, (255, 0, 255)) #方法，作用：在图像上打印文字，设置字体，颜色，大小
#         else :
#             cv2.putText(frame, "OPEN", (75, 250), font, 7, (0, 255, 0))
#             start = time.time()#记时
#         print("闭眼时间:%.2f秒"%(end-start))#获取睁闭眼时间差
#
#         #判断是否疲劳
#         if (end-start) > 2 :
#             cv2.putText(frame, "TIRED", (200, 325), font, 7, (0, 0, 255))
#             duration = 1000
#             freq = 1000
#             winsound.Beep(freq, duration)#调用喇叭，设置声音大小，与时间长短
#
#     cv2.imshow("Frame", frame)#方法，作用，创建一个窗口，将画面投影到窗口中
#
#     #推出键设置，按Ese键退出
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# #释放窗口，关闭摄像头
# cap.release()
# cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------
#
# from __future__ import division
# import dlib
# import cv2
#
#
# img = cv2.imread("E:\\pyitem\\opencv_img\\photo\\me.jpg")
#
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
# detector = dlib.get_frontal_face_detector()
#
# dets = detector(gray_img, 1)
#
# # detector = dlib.get_frontal_face_detector(gray_img, 1)
#
# # 使用模型构建特征提取器
# predictor = dlib.shape_predictor('E:\\pyitem\\venv\Lib\site-packages\\shape_predictor_68_face_landmarks.dat')
#
# for i, d in enumerate(dets):
# # for i, d in enumerate(detector):
#     # 使用predictor进行人脸关键点检测 shape为返回的结果
#     shape = predictor(gray_img, d)
#
#     for index, pt in enumerate(shape.parts()):
#         print('Part {}: {}'.format(index, pt))
#         pt_pos = (pt.x, pt.y)
#         cv2.circle(img, pt_pos, 1, (255, 0, 0), 2)
#         # 利用cv2.putText标注序号
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, str(index + 1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
#
# cv2.imshow('img', img)
# k = cv2.waitKey(0)
# cv2.destroyAllWindows()

# --------------------------人脸检测(可正常使用)------------------------------------------
import cv2


def face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier('E:\\pyitem\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale2(gray, 1.02, 5, minSize=(20, 20))
    # print(faces[0])
    # print(faces)
    for face in faces:
        for x, y, w, h in faces[0]:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('res', img)
    # cv2.imshow('res', img)


if __name__ == '__main__':
    #img = cv2.imread('E:\\pyitem\\opencv_img\\photo\\me.jpg')
    # # img = cv2.resize(img, (600, 300))
    # cv2.imshow('original', img)
    # face_detect(img)
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = capture.read()
        face_detect(frame)
        if cv2.waitKey(20) & 0xff == 27:
            break
    capture.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------------
# import cv2
# import os
# # 定义文件夹所在路径
# image_files_path = 'E:\\pyitem\\opencv_img\\photo'
# # image_files = os.path.join(image_files_path, '.jpg')
# # 读取该文件夹下所有文件名
# image_files = os.listdir(image_files_path)
# for file in image_files:
#     img = cv2.imread(os.path.join(image_files_path, file))
#     # 另一种方式读取
#     # img = cv2.imread(image_files_path+'\\'+file)
#     # 检测人脸
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     face_detector = cv2.CascadeClassifier(
#         'E:\\pyitem\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt_tree.xml')
#     faces = face_detector.detectMultiScale2(gray, 1.02, 5, minSize=(20, 20))
#     print(faces)
#     # if faces != ((), ()):
#     #     for x, y, w, h in faces[0]:
#     #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     #     cv2.imshow('image', img)
#     #     cv2.waitKey(0)
#     #     cv2.destroyAllWindows()
