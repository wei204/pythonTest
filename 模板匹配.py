import cv2


# 模板查找函数
def template():
    # 读入模板图片
    tpl = cv2.imread('head.jpg')
    tpl = cv2.resize(tpl, (40, 44))    # 宽高
    print(tpl.shape)
    # cv2.imshow('head1', tpl)
    # img = cv2.imread('me.jpg')
    # tpl = img[220:270, 130:180]
    # cv2.imshow('me_temp', tpl)
    # 读入待检测目标图片
    target = cv2.imread('heying.jpg')
    # 匹配方式
    methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED,
               cv2.TM_CCOEFF_NORMED]
    # methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
               #  相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好
               # 归一化相关系数匹配法
               # 归一化平方差匹配法
    # 获取模板高度，宽度
    h, w = tpl.shape[:2]
    for i in methods:
        print(i)
        res = cv2.matchTemplate(target, tpl, i)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    # 获取最大最小值及其索引位置
        if i == cv2.TM_SQDIFF_NORMED or i == cv2.TM_SQDIFF:        # 当使用匹配方法为cv2.TM_SQDIFF 与cv2.TM_SQDIFF_NORMED时，最小值对应匹配位置
            tl = min_loc                   # tl为匹配位置，即左上角位置。tl[0],tl[1]分别为行、列。
            print(tl)
        else:
            tl = max_loc
        # 在匹配位置绘制矩形
        tr = (tl[0]+w, tl[1]+h)      # tr 为匹配位置右下角的点
        cv2.rectangle(target, tl, tr, (0, 0, 255), 2)
        cv2.imshow('methods-'+str(i), target)

# def template():
#     # 读入模板图片
#     tpl = cv2.imread('head.jpg')
#     tpl = cv2.resize(tpl, (40, 44))
#     cv2.imshow('head', tpl)
#     print(tpl.shape)
#     # img = cv2.imread('me.jpg')
#     # tpl = img[220:270, 130:180]
#     # cv2.imshow('me_temp', tpl)
#     # 读入待检测目标图片
#     target = cv2.imread('heying.jpg')
#     # 匹配方式
#     methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED,
#                cv2.TM_CCOEFF_NORMED]
#     # methods = [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]
#                #  相关匹配法：该方法采用乘法操作；数值越大表明匹配程度越好
#                # 归一化相关系数匹配法
#                # 归一化平方差匹配法
#     # 获取模板高度，宽度
#     h, w = tpl.shape[:2]
#     # for i in methods:
#     #     print(i)
#     #     res = cv2.matchTemplate(target, tpl, i)
#     #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)    # 获取最大最小值及其索引位置
#     #     if i == cv2.TM_SQDIFF_NORMED or i == cv2.TM_SQDIFF:        # 当使用匹配方法为cv2.TM_SQDIFF 与cv2.TM_SQDIFF_NORMED时，最小值对应匹配位置
#     #         tl = min_loc                   # tl为匹配位置，即左上角位置。tl[0],tl[1]分别为行、列。
#     #         print(tl)
#     #     else:
#     #         tl = max_loc
#     #     # 在匹配位置绘制矩形
#     #     tr = (tl[0]+w, tl[1]+h)      # tr 为匹配位置右下角的点
#     #     cv2.rectangle(target, tl, tr, (0, 0, 255), 2)
#     #     cv2.imshow('methods-'+str(i), target)
#     res = cv2.matchTemplate(target, tpl, cv2.TM_CCORR_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#     tl = max_loc
#     tr = (tl[0]+w, tl[1]+h)
#     cv2.rectangle(target, tl, tr, (0, 0, 255), 2)
#     cv2.imshow('methods-3', target)



if __name__ == '__main__':
    img1 = cv2.imread('head1.jpg')
    # print(img1.shape)
    template()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
