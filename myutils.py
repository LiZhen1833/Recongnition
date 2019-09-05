import cv2
import numpy as np
from imutils import contours


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] #用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def template_img_process(img):
    ref=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv_show("ref",ref)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1] #binary
    # cv_show("thresh",ref)

    #计算轮廓
    ref_,refCnts,hierarchy=cv2.findContours(ref.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    refCnts=sort_contours(refCnts,method="left-to-right")[0]  #sort

    digits = {}

    # 遍历每一个轮廓
    for (i, c) in enumerate(refCnts):
        # 计算外接矩形并且resize成合适大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        # cv_show("img",roi)
        # 每一个数字对应每一个模板
        digits[i] = roi

    return digits

def card_img_process(img, digits):
    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


    img = resize(img, width=300)
    cv_show("img",img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show("gray",gray)
    #顶帽变换
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    cv_show('thresh', tophat)
    #Sobel 边缘检测
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,  # ksize=-1相当于用3*3的
                      ksize=-1)

    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    #形态学操作
    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    # THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv_show("thresh",thresh)

    #计算轮廓
    thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = img.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    cv_show("cur_img",cur_img)

    locs = []
    # 遍历轮廓(筛选)
    for (i, c) in enumerate(cnts):
        # 计算矩形
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if ar > 2.5 and ar < 4.0:

            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # 符合的留下来
                locs.append((x, y, w, h))

    #对 locs 进行排序，按照x的大小进行排序
    locs=sorted(locs,key=lambda x:x[0]) #这里的x不代表x坐标，而是（x,y,w,h）向量的指代
    output=[]

    for (i,(gX,gY,gW,gH))in enumerate(locs):
        groupOutput=[]
        # 根据坐标提取每一个组
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        cv_show("group",group)
        group=cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        cv_show("group", group)
        #计算轮廓
        group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = contours.sort_contours(digitCnts,method="left-to-right")[0]

        # 计算每一组中的每一个数值
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的的大小
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            # cv_show("roi", roi)
            roi = cv2.resize(roi, (57, 88))
            cv_show("roi",roi)

            # 计算匹配得分
            scores = []

            # 在模板中计算每一个得分
            for (digit, digitROI) in digits.items():
                # 模板匹配
                result = cv2.matchTemplate(roi, digitROI,
                                           cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score) #记录匹配得分

            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))
        # 画出来
        cv2.rectangle(img, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(img, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        # 得到结果
        output.extend(groupOutput) #添加一个序列

    # 打印结果
    print("Credit Card #: {}".format("".join(output)))
    cv2.imshow("Image", img)
    cv2.waitKey(0)
