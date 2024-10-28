import cv2                          # opencv    读取图像默认为BGR
import matplotlib.pyplot as plt     # matplotlib显示图像默认为RGB
import numpy as np


def sort_contours(cnt_s, method="left-to-right"):
    reverse = False
    ii_myutils = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        ii_myutils = 1
    bounding_boxes = [cv2.boundingRect(cc_myutils) for cc_myutils in cnt_s]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnt_s, bounding_boxes) = zip(*sorted(zip(cnt_s, bounding_boxes), key=lambda b: b[1][ii_myutils], reverse=reverse))
    return cnt_s, bounding_boxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim_myutils = None
    (h_myutils, w_myutils) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r_myutils = height / float(h_myutils)
        dim_myutils = (int(w_myutils * r_myutils), height)
    else:
        r_myutils = width / float(w_myutils)
        dim_myutils = (width, int(h_myutils * r_myutils))
    resized = cv2.resize(image, dim_myutils, interpolation=inter)
    return resized


def extract_template(image):
    """（1）提取模板图像中每个数字（0~9）"""
    ref_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
    ref_BINARY = cv2.threshold(ref_gray, 10, 255, cv2.THRESH_BINARY_INV)[1]  # 转换成二值图像
    refCnts, hierarchy = cv2.findContours(ref_BINARY.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_Contours = image.copy()
    cv2.drawContours(img_Contours, refCnts, -1, (0, 0, 255), 3)

    # 绘制图像
    plt.subplot(221), plt.imshow(image, 'gray'), plt.title('(0)ref')
    plt.subplot(222), plt.imshow(ref_gray, 'gray'), plt.title('(1)ref_gray')
    plt.subplot(223), plt.imshow(ref_BINARY, 'gray'), plt.title('(2)ref_BINARY')
    plt.subplot(224), plt.imshow(img_Contours, 'gray'), plt.title('(3)img_Contours')
    plt.show()

    # 排序所有轮廓（从左到右，从上到下）
    refCnts = sort_contours(refCnts, method="left-to-right")[0]

    # 提取所有轮廓
    digits = {}  # 保存每个模板的数字
    for (index, region) in enumerate(refCnts):
        (x, y, width, height) = cv2.boundingRect(region)  # 得到轮廓(数字)的外接矩形的左上角的(x, y)坐标、宽度和长度
        roi = ref_BINARY[y:y + height, x:x + width]  # 获得外接矩形的坐标
        roi = cv2.resize(roi, (57, 88))  # 将感兴趣区域的图像(数字)resize相同的大小
        digits[index] = roi

    """#################################################################################################################
    # 函数功能：用于在二值图像中轮廓检测
    # 函数说明：contours, hierarchy = cv2.findContours(image, mode, method)
    # 参数说明：
    #         image：        输入的二值图像，通常为灰度图像或者是经过阈值处理后的图像。
    #         mode：         轮廓检索模式，指定轮廓的层次结构。
    #                             cv2.RETR_EXTERNAL：    仅检测外部轮廓。
    #                             cv2.RETR_LIST：        检测所有轮廓，不建立轮廓层次。
    #                             cv2.RETR_CCOMP：       检测所有轮廓，将轮廓分为两级层次结构。
    #                             cv2.RETR_TREE：        检测所有轮廓，建立完整的轮廓层次结构。
    #         method：       轮廓逼近方法，指定轮廓的近似方式。
    #                             cv2.CHAIN_APPROX_NONE：        存储所有的轮廓点，相邻的点之间不进行抽稀。
    #                             cv2.CHAIN_APPROX_SIMPLE：      仅存储轮廓的端点，相邻的点之间进行抽稀，以减少存储容量。
    #                             cv2.CHAIN_APPROX_TC89_L1：     使用 Teh-Chin 链逼近算法中的 L1 范数，以减少存储容量。
    #                             cv2.CHAIN_APPROX_TC89_KCOS：   使用 Teh-Chin 链逼近算法中的 KCOS 范数，以减少存储容量。
    # 输出参数：
    #         contours：     检测到的轮廓列表，每个轮廓是一个点的列表。
    #         hierarchy：    轮廓的层次结构，一般用不到，可以忽略。
    #
    # 备注1：轮廓就是将连续的点（连着边界）连在一起的曲线，具有相同的颜色或者灰度。轮廓在形状分析和物体的检测和识别中很有用。
    # 备注2：函数在opencv2只返回两个值：contours, hierarchy。
    # 备注3：函数在opencv3会返回三个值：image, countours, hierarchy
    #################################################################################################################"""

    """#################################################################################################################
    # 函数功能：用于在图像上绘制轮廓。
    # 函数说明：cv2.drawContours(image, contours, contourIdx, color, thickness=1, lineType=cv2.LINE_8, hierarchy=None, maxLevel=None, offset=None)
    # 参数说明：
    #         image：        要绘制轮廓的图像，可以是单通道或多通道的图像。
    #         contours：     要绘制的轮廓列表，通常通过 cv2.findContours 函数获取。
    #         contourIdx：   要绘制的轮廓索引，如果为负数则绘制所有轮廓。
    #         color：        轮廓的颜色，通常是一个元组，如 (0, 255, 0) 表示绿色。
    #         thickness：    轮廓线的粗细，如果为负数或 cv2.FILLED 则填充轮廓区域。
    #         lineType：     线的类型
    #                             cv2.LINE_4：4连接线。
    #                             cv2.LINE_8：8连接线。
    #                             cv2.LINE_AA：抗锯齿线。
    #         hierarchy：    轮廓的层级结构，一般由 cv2.findContours 函数返回，可选参数。
    #         maxLevel：     要绘制的轮廓的最大层级，一般由 cv2.findContours 函数返回，可选参数。
    #         offset：       轮廓偏移量，一般不需要设置，可选参数。
    #
    # 备注：使用copy()复制图像后绘制, 否则原图会同时改变。
    #################################################################################################################"""
    return digits


def extract_card(image):
    """（2）提取银行卡的所有轮廓"""
    rect_Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    square_Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    """#################################################################################################################
    # 函数功能：用于创建指定形状和大小的结构元素（structuring element） ———— 常用于图像形态学处理中的腐蚀、膨胀、开运算、闭运算等操作。
    # 函数说明：element = cv2.getStructuringElement(shape, ksize)
    # 参数说明：		
    #         shape：    结构元素的形状：
    #                         cv2.MORPH_RECT：   矩形结构元素。
    #                         cv2.MORPH_CROSS：  十字形结构元素。
    #                         cv2.MORPH_ELLIPSE：椭圆形结构元素。
    #         ksize：    结构元素的卷积核大小，通常是一个元组 (width, height)。如：(3, 3)表示3*3的卷积核
    #################################################################################################################"""

    image_tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, rect_Kernel)  # 礼帽运算 ———— 用于突出亮区域
    """#################################################################################################################
    # 函数功能：用于对图像进行形态学操作，如腐蚀、膨胀、开运算、闭运算等。
    # 函数说明：dst = cv2.morphologyEx(src, op, kernel)
    # 参数说明：
    #         src：      输入图像，通常是灰度图像或二值图像。
    #         op：       形态学操作类型，可以是以下几种之一：
    #                             cv2.MORPH_ERODE：      腐蚀操作。
    #                             cv2.MORPH_DILATE：     膨胀操作。
    #                             cv2.MORPH_OPEN：       开运算（先腐蚀后膨胀）。            开运算可以用来消除小黑点。
    #                             cv2.MORPH_CLOSE：      闭运算（先膨胀后腐蚀）。            闭运算可以用来突出边缘特征。
    #                             cv2.MORPH_GRADIENT：   形态学梯度（膨胀图像减去腐蚀图像）。  突出团块(blob)的边缘，保留物体的边缘轮廓。
    #                             cv2.MORPH_TOPHAT：     顶帽运算（原始图像减去开运算图像）。  突出比原轮廓亮的部分。
    #                             cv2.MORPH_BLACKHAT：   黑帽运算（闭运算图像减去原始图像）。  突出比原轮廓暗的部分。
    #         kernel：   结构元素，用于指定形态学操作的形状和大小。
    #################################################################################################################"""

    """#################################################################################################################
    # 函数功能：Sobel算子是一种常用的边缘检测算子 ———— 对噪声具有平滑作用，提供较为精确的边缘方向信息，但是边缘定位精度不够高。
    # 操作步骤：    
    #             备注1：分别计算x和y，再求和（效果好），若同时对x和y进行求导，会导致部分信息丢失。（不建议）
    #             备注2：对于二维图像，Sobel算子在x,y两个方向求导，故有不同的两个卷积核（Gx, Gy），且Gx的转置等于Gy。分别反映每个像素在水平和在垂直方向上的亮度变换情况.
    #             边缘即灰度像素值快速变化的地方。如：黑到白的边界
    # 
    # 函数说明：dst = cv2.Sobel(src, ddepth, dx, dy, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    # 参数说明：	
    #             src:        输入图像，通常是单通道的灰度图像。通常情况下，输入图像的深度是 cv2.CV_8U（8 位无符号整型）
    #             ddepth:     输出图像的深度，通常为-1表示与输入图像相同。支持以下类型：cv2.CV_16S、cv2.CV_32F、cv2.CV_64F
    #             dx:          x 方向上的导数阶数，0 表示不对 x 方向求导数。
    #             dy:          y 方向上的导数阶数，0 表示不对 y 方向求导数。
    #             ksize:       滤波器的内核大小。必须是 -1、1、3、5 或 7，表示内核的大小是 1x1、3x3、5x5 或 7x7。 
    #                                       -1表示使用 Scharr 滤波器。Scharr 滤波器是 Sobel 滤波器的改进版本，具有更好的性能。
    #             scale:       （可选）表示计算结果的缩放因子。
    #             delta:       （可选）表示输出图像的偏移值。
    #             borderType:  （可选）表示边界处理方式。
    # 返回参数：
    #             滤波后图像
    #################################################################################################################"""

    """#################################################################################################################
    # 函数功能：用于将输入数组的每个元素乘以缩放因子 alpha，然后加上偏移量 beta，并取结果的绝对值后转换为无符号8位整数类型。
    # 函数说明：dst = cv2.convertScaleAbs(src, alpha=1, beta=0)
    # 参数说明：
    #         src：      输入数组，可以是任意类型的数组。
    #         alpha：    缩放因子
    #         beta：     偏移量
    # 返回参数：
    #         dst       转换后数组（与输入数组相同大小和类型的数组，数据类型为无符号8位整数。）
    #################################################################################################################"""
    image_gradx = cv2.Sobel(image_tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    image_gradx = np.absolute(image_gradx)  # 求绝对值
    (minVal, maxVal) = (np.min(image_gradx), np.max(image_gradx))  # 计算最大最小边界差值
    image_gradx = (255 * ((image_gradx - minVal) / (maxVal - minVal)))  # 归一化处理（0~1）
    image_gradx = image_gradx.astype("uint8")  # 数据类型转换

    """
    # 分别计算x和y，再求和（效果好）
    sobel_Gx1 = cv2.Sobel(image_tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    sobel_Gx_Abs1 = cv2.convertScaleAbs(sobel_Gx1)  # （1）左边减右边（2）白到黑是正数，黑到白就是负数，且所有的负数会被截断成0，所以要取绝对值。
    sobel_Gy1 = cv2.Sobel(image_tophat, cv2.CV_64F, 0, 1, ksize=3)
    sobel_Gy_Abs1 = cv2.convertScaleAbs(sobel_Gy1)
    sobel_Gx_Gy_Abs1 = cv2.addWeighted(sobel_Gx_Abs1, 0.5, sobel_Gy_Abs1, 0.5, 0)   # 权重值x + 权重值y +偏置b
    """

    """#################################################################################################################
    # 函数说明：全局阈值分割 ———— 根据阈值将图像的像素分为两个类别：高于阈值的像素和低于阈值的像素。
    # 函数说明：ret, dst = cv2.threshold(src, thresh, max_val, type)
    # 参数说明：    	
    #         src:   	    输入灰度图像
    #         thresh: 	    阈值
    #         max_val:    	阈值上限。通常为255（8-bit）。
    #         type:   	    二值化操作的类型，包含以下5种类型：
    #                               (1) cv2.THRESH_BINARY             超过阈值部分取max_val（最大值），否则取0
    #                               (2) cv2.THRESH_BINARY_INV         THRESH_BINARY的反转
    #                               (3) cv2.THRESH_TRUNC              大于阈值部分设为阈值，否则不变
    #                               (4) cv2.THRESH_TOZERO             大于阈值部分不改变，否则设为0
    #                               (5) cv2.THRESH_TOZERO_INV         THRESH_TOZERO的反转
    # 返回参数：     
    #         ret           浮点数，表示最终使用的阈值。
    #         dst   	    经过阈值分割操作后的二值图像
    #################################################################################################################"""
    # 闭运算（先膨胀，再腐蚀）———— 将银行卡分成四个部分，每个部分的四个数字连在一起
    image_CLOSE = cv2.morphologyEx(image_gradx, cv2.MORPH_CLOSE, square_Kernel)
    image_thresh = cv2.threshold(image_CLOSE, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 阈值分割
    # （二次）闭运算 ———— 将四个连在一起的数字进行填充形成一个整体。
    image_2_dilate = cv2.dilate(image_thresh, square_Kernel, iterations=2)  # 膨胀(迭代次数2次)
    image_1_erode = cv2.erode(image_2_dilate, square_Kernel, iterations=1)  # 腐蚀(迭代次数1次)
    image_2_CLOSE = image_1_erode
    threshCnts, hierarchy = cv2.findContours(image_2_CLOSE.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测
    image_Contours = image_resize.copy()
    cv2.drawContours(image_Contours, threshCnts, -1, (0, 0, 255), 3)  # 绘制轮廓
    # 绘制图像
    plt.subplot(241), plt.imshow(image, 'gray'), plt.title('(0)image_card')
    plt.subplot(242), plt.imshow(image_gray, 'gray'), plt.title('(1)image_gray')
    plt.subplot(243), plt.imshow(image_tophat, 'gray'), plt.title('(2)image_tophat')
    plt.subplot(244), plt.imshow(image_gradx, 'gray'), plt.title('(3)image_gradx')
    plt.subplot(245), plt.imshow(image_CLOSE, 'gray'), plt.title('(4)image_CLOSE')
    plt.subplot(246), plt.imshow(image_thresh, 'gray'), plt.title('(5)image_thresh')
    plt.subplot(247), plt.imshow(image_2_CLOSE, 'gray'), plt.title('(6)image_2_CLOSE')
    plt.subplot(248), plt.imshow(image_Contours, 'gray'), plt.title('(7)image_Contours')
    plt.show()

    return threshCnts


def extract_digits(image_gray, threshCnts, digits):
    """3311、识别出四个数字一组的所有轮廓（理论上是四个）"""
    locs = []  # 保存四个数字一组的轮廓坐标
    # 遍历轮廓
    for (index, region) in enumerate(threshCnts):
        (x, y, wight, height) = cv2.boundingRect(region)  # 计算矩形
        ar = wight / float(height)  # （四个数字一组）的长宽比
        # 匹配(四个数字为一组)轮廓的大小 ———— 根据实际图像调整
        if 2.0 < ar < 4.0:
            if (35 < wight < 60) and (10 < height < 20):
                locs.append((x, y, wight, height))
    locs = sorted(locs, key=lambda x: x[0])  # 排序（从左到右）

    """3322、在四个数字一组中，提取每个数字的轮廓坐标并进行模板匹配"""
    output_all_group = []  # 保存（所有组）匹配结果
    for (ii, (gX, gY, gW, gH)) in enumerate(locs):
        """遍历银行卡的每个组（理论上是四组）"""
        image_group = image_gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]  # 坐标索引每个组图像，并将每个轮廓的结果放大一些，避免信息丢失
        image_group_threshold = cv2.threshold(image_group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化处理
        digitCnts, hierarchy = cv2.findContours(image_group_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓
        digitCnts_sort = sort_contours(digitCnts, method="left-to-right")[0]  # 排序

        """遍历每个组中的每个数字（理论上每个组是四个数字）"""
        output_group = []  # 保存（每个组）匹配结果
        for jj in digitCnts_sort:
            (x, y, wight, height) = cv2.boundingRect(jj)  # 获取数字的轮廓
            image_digit = image_group[y:y + height, x:x + wight]  # 获取数字的坐标
            image_digit = cv2.resize(image_digit, (57, 88))  # 图像缩放（保持与模板图像中的数字图像一致）
            cv2.imshow("Image", image_digit)
            cv2.waitKey(200)  # 延迟200ms

            """遍历模板图像的每个数字，并计算（轮廓数字）和（模板数字）匹配分数（理论上模板的十个数字））"""
            scores = []  # 计算匹配分数：（轮廓中的数字）和（模板中的数字）
            for (kk, digitROI) in digits.items():
                result = cv2.matchTemplate(image_digit, digitROI, cv2.TM_CCOEFF)
                (_, max_score, _, _) = cv2.minMaxLoc(result)  # max_score表示最大值
                scores.append(max_score)
            output_group.append(str(np.argmax(scores)))  # 将最大匹配分数对应的数字保存下来

            """#########################################################################################################
            # 函数功能：模板匹配 ———— 在输入图像上滑动目标模板，并计算目标模板与图像的匹配程度来实现目标检测。
            # 函数说明：cv2.matchTemplate(image, template, method)
            # 参数说明：
            #         image：      输入图像，需要在其中搜索目标模板。
            #         templ：      目标模板，需要在输入图像中匹配的部分。
            #         method：     匹配方法
            #                             cv2.TM_SQDIFF：         计算平方差。         计算结果越接近0，越相关
            #                             cv2.TM_CCORR：          计算相关性。          计算结果越大，越相关
            #                             cv2.TM_CCOEFF：         计算相关系数。         计算结果越大，越相关
            #                             cv2.TM_SQDIFF_NORMED：  计算（归一化）平方差。   计算结果越接近0，越相关
            #                             cv2.TM_CCORR_NORMED：   计算（归一化）相关性。   计算结果越接近1，越相关
            #                             cv2.TM_CCOEFF_NORMED：  计算（归一化）相关系数。  计算结果越接近1，越相关
            #                      备注：归一化效果更好
            #########################################################################################################"""

            """#########################################################################################################
            # 函数功能：用于找到数组中的最小值和最大值，以及它们的位置。
            # 函数说明：min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ret)
            # 输入参数：
            #         ret       cv2.matchTemplate函数返回的矩阵，即匹配结果图像。
            # 返回参数：
            #         min_val：匹配结果图像中的最小值。
            #         max_val：匹配结果图像中的最大值。
            #         min_loc：最小值对应的位置（x，y）。
            #         max_loc：最大值对应的位置（x，y）。
            #
            # 备注：如果模板匹配方法是平方差或者归一化平方差，则使用min_loc; 否则使用max_loc
            #########################################################################################################"""

        # 在绘制矩形 ———— 在原图上，用矩形将" 四个数字一组 "画出来（理论上共有四个矩形，对应四个组）
        cv2.rectangle(image_resize, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image_resize, "".join(output_group), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        output_all_group.extend(output_group)
        """#########################################################################################################
        # 函数功能：用于在图像上绘制文本。———— 支持在图像的指定位置添加文字，并设置文字的字体、大小、颜色等属性。
        # 函数说明：cv2.putText(img, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_AA, bottomLeftOrigin=False)
        # 参数说明：
        #         img：      要绘制文本的图像。
        #         text：     要绘制的文本内容。
        #         org：      文本的起始坐标，即左下角的坐标。
        #         fontFace： 字体类型，可以是以下几种之一：
        #                     cv2.FONT_HERSHEY_SIMPLEX：正常尺寸的简单字体。
        #                     cv2.FONT_HERSHEY_PLAIN：小号字体。
        #                     cv2.FONT_HERSHEY_DUPLEX：正常尺寸的复杂字体。
        #                     cv2.FONT_HERSHEY_COMPLEX：正常尺寸的复杂字体。
        #                     cv2.FONT_HERSHEY_TRIPLEX：双倍大小的复杂字体。
        #                     cv2.FONT_HERSHEY_COMPLEX_SMALL：小号复杂字体。
        #                     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX：手写风格的简单字体。
        #                     cv2.FONT_HERSHEY_SCRIPT_COMPLEX：手写风格的复杂字体。
        #         fontScale：字体大小缩放比例。
        #         color：    文本颜色，通常是一个元组，如 (255, 0, 0) 表示蓝色。
        #         thickness：文本轮廓线的粗细，默认为 1。
        #         lineType： 文本的线型，可以是以下几种之一：
        #                     cv2.LINE_AA：抗锯齿线。
        #                     cv2.LINE_4：4连接线。
        #                     cv2.LINE_8：8连接线。
        #         bottomLeftOrigin：可选参数，表示坐标原点是否在左下角，默认为 False，即左上角。
        # 返回参数：
        #         无，直接在输入图像上绘制了文本。
        #########################################################################################################"""
    cv2.imshow("Image", image_resize)
    cv2.waitKey(0)


if __name__ == "__main__":
    """11、提取模板图像中每个数字模板（0~9）"""
    image_template = cv2.imread(r'template.png')
    digits = extract_template(image_template)

    """22、提取信用卡的所有轮廓（需要根据实际情况调整）"""
    image_card = cv2.imread(r'card.png')
    image_resize = resize(image_card, width=300)
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    threshCnts = extract_card(image_card)

    """33、提取银行卡" 四个数字一组 "轮廓，然后每个轮廓与模板的每一个数字进行匹配，得到最大匹配结果，并在原图上绘制结果"""
    extract_digits(image_gray, threshCnts, digits)

