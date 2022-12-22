import cv2
import numpy as np


def ROI_correct(roi):
    # 对ROI区域进行校正
    ROTATED_SIZE_W = lsPointsChoose[1][0] - lsPointsChoose[0][0]  # 透视变换后的表盘图像大小
    ROTATED_SIZE_H = lsPointsChoose[3][1] - lsPointsChoose[0][1]  # 透视变换后的表盘图像大小
    # 原图中书本的四个角点(左上、右上、右下、左下),与变换后矩阵位置
    pts1 = np.float32([lsPointsChoose[0], lsPointsChoose[1], lsPointsChoose[2], lsPointsChoose[3]])
    # pts1 = np.float32([[20, 195], [963, 223], [959, 560], [14, 615]])
    # 变换后矩阵位置
    pts2 = np.float32([[0, 0], [ROTATED_SIZE_W, 0], [ROTATED_SIZE_W, ROTATED_SIZE_H], [0, ROTATED_SIZE_H]])
    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    roi_correct = cv2.warpPerspective(roi, M, (ROTATED_SIZE_W, ROTATED_SIZE_H))
    # cv2.imshow('roi_correct', roi_correct)
    cv2.imwrite('roi_correct.jpg', roi_correct)
    return roi_correct

def on_mouse1(event, x, y, flags, params):
    global num
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        num = num + 1
        # num_circle.append(num)
        # print('pointsCount:', pointsCount)
        p1 = (x, y)
        #将选中的点保存到列表result_point中
        # result_point.append([x, y])
        # print(x, y)
        # 画出点击的点
        cv2.circle(result_img, p1, 1, (0, 255, 0), 2)
        cv2.imshow('circle_detection', result_img)
    # #双击消除选中的点 待完善
    # if event == cv2.EVENT_LBUTTONDBLCLK:
    #     p2 = (x, y)
    #     num = num - 1




def detection_circle(roi_correct):
    # 对ROI区域进行检测
    # result_img = result_img.copy()
    gray = cv2.cvtColor(roi_correct, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('circle_detection')

    # 添加进度条
    def nothing(x):
        pass

    cv2.createTrackbar('minRadius', 'circle_detection', 1, 15, nothing)
    cv2.createTrackbar('maxRadius', 'circle_detection', 15, 20, nothing)
    while (1):
        gray1 = gray.copy()
        minRadius = cv2.getTrackbarPos('minRadius', 'circle_detection')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'circle_detection')
        circles = cv2.HoughCircles(gray1, cv2.HOUGH_GRADIENT, 1, 15, param1=50, param2=10, minRadius=minRadius,
                                   maxRadius=maxRadius)
        circles = np.uint16(np.around(circles))
        # print('圆的个数为：', len(circles[0]))
        global result_img
        result_img = roi_correct.copy()
        for i in circles[0]:
            # cv2.circle(result_img, (i[0], i[1]), i[2], (0, 0, 255), 2)
            cv2.circle(result_img, (i[0], i[1]), 1, (0, 0, 255), 2)
        cv2.imshow('circle_detection', result_img)
        #
        if cv2.waitKey(1) == ord('q'):
            break
    # 手动添加漏检的钢管
    cv2.setMouseCallback('circle_detection', on_mouse1)
    cv2.imshow('circle_detection', result_img)
    cv2.waitKey(0)
    print('钢管数量：',len(circles[0]) + num)



    # print('圆的个数为：', len(circles[0]))



if __name__ == '__main__':
    # -----------------------鼠标操作相关------------------------------------------
    lsPointsChoose = []
    tpPointsChoose = []
    pointsCount = 0
    count = 0
    pointsMax = 6

    # num_circle = []
    # result_point = []
    num = 0
    # 鼠标事件
    def on_mouse(event, x, y, flags, param):
        global img, point1, point2, count, pointsMax
        global lsPointsChoose, tpPointsChoose  # 存入选择的点
        global pointsCount  # 对鼠标按下的点计数
        global img2, ROI_bymouse_flag
        img2 = img.copy()  # 此行代码保证每次都重新再原图画  避免画多了
        # -----------------------------------------------------------
        #    count=count+1
        #    print("callback_count",count)
        # --------------------------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            pointsCount = pointsCount + 1
            print('pointsCount:', pointsCount)
            point1 = (x, y)
            print(x, y)
            # 画出点击的点
            cv2.circle(img2, point1, 10, (0, 255, 0), 2)

            # 将选取的点保存到list列表里
            lsPointsChoose.append([x, y])  # 用于转化为darry 提取多边形ROI
            tpPointsChoose.append((x, y))  # 用于画点
            # ----------------------------------------------------------------------
            # 将鼠标选的点用直线连起来
            print(len(tpPointsChoose))
            for i in range(len(tpPointsChoose) - 1):
                print('i', i)
                cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
            # ----------------------------------------------------------------------
            # ----------点击到pointMax时可以提取去绘图----------------

            cv2.imshow('src', img2)

        # -------------------------右键按下清除轨迹-----------------------------
        if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击
            print("right-mouse")
            pointsCount = 0
            tpPointsChoose = []
            lsPointsChoose = []
            print(len(tpPointsChoose))
            for i in range(len(tpPointsChoose) - 1):
                print('i', i)
                cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)
            cv2.imshow('src', img2)

        # -------------------------双击 结束选取-----------------------------
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # -----------绘制感兴趣区域-----------
            ROI_byMouse()
            ROI_bymouse_flag = 1
            lsPointsChoose = []


    def ROI_byMouse():
        # global src, ROI, ROI_flag, mask2
        global mask2
        mask = np.zeros(img.shape, np.uint8)
        pts = np.array([lsPointsChoose], np.int32)  # pts是多边形的顶点列表（顶点集）
        pts = pts.reshape((-1, 1, 2))
        # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
        # OpenCV中需要先将多边形的顶点坐标变成顶点数×1×2维的矩阵，再来绘制

        # --------------画多边形---------------------
        mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
        ##-------------填充多边形---------------------
        mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
        # cv2.imshow('mask', mask2)
        # cv2.imwrite('photo/mask.jpg', mask2)
        contours, hierarchy = cv2.findContours(cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        roi = cv2.bitwise_and(mask2, img)
        cv2.imwrite('roi.jpg', roi)
        cv2.imshow('roi', roi)
        # 对ROI区域进行校正
        roi_correct = ROI_correct(roi)
        # # 检测ROI区域钢管数量
        detection_circle(roi_correct)
        # result_img, circle = detection_circle(roi_correct)
        # cv2.setMouseCallback('circle_detection', on_mouse1)
        # cv2.imshow('circle_detection', result_img)
        # print('钢管数量：',circle + len(num_circle))



    img = cv2.imread('../photo/gangguan.jpg')
    #选择ROI区域
    cv2.namedWindow('src')
    cv2.setMouseCallback('src', on_mouse)
    cv2.imshow('src', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()