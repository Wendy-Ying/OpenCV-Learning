import cv2
import numpy as np
"""#########################################################
# 预定义框架说明
# 定义一个Stitcher类：stitch()、detectAndDescribe()、matchKeypoints()、drawMatches()
#           stitch()                拼接函数
#           detectAndDescribe()     检测图像的SIFT关键特征点，并计算特征描述子
#           matchKeypoints()        匹配两张图片的所有特征点
#           cv2.findHomography()    计算单映射变换矩阵
#           cv2.warpPerspective()   透视变换（作用：缝合图像）
#           drawMatches()           建立直线关键点的匹配可视化
#
# 备注：cv2.warpPerspective()需要与cv2.findHomography()搭配使用。
#########################################################"""


class Stitcher:
    ##################################################################################
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        (imageB, imageA) = images                               # 获取输入图片
        (kpsA, featuresA) = self.detectAndDescribe(imageA)      # 检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)    # 匹配两张图片的所有特征点，返回匹配结果。

        if M is None:       # 如果返回结果为空，没有匹配成功的特征点，退出算法
            return None

        # 否则，提取匹配结果 #
        (matches, H, status) = M     # H是3x3视角变换矩阵
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))   # 将图片A进行视角变换，result是变换后图片
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB    # 将图片B传入result图片最左端

        if showMatches:     # 检测是否需要显示图片匹配
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)     # 生成匹配图片
            return (result, vis)

        return result

    ##################################################################################
    def detectAndDescribe(self, image):
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            # 将彩色图片转换成灰度图
        descriptor = cv2.xfeatures2d.SIFT_create()                  # 建立SIFT生成器
        """#####################################################
        # 如果是OpenCV3.X，则用cv2.xfeatures2d.SIFT_create方法来实现DoG关键点检测和SIFT特征提取。
        # 如果是OpenCV2.4，则用cv2.FeatureDetector_create方法来实现关键点的检测（DoG）。
        #####################################################"""
        (kps, features) = descriptor.detectAndCompute(image, None)  # 检测SIFT特征点，并计算描述子
        kps = np.float32([kp.pt for kp in kps])                     # 将结果转换成NumPy数组

        return (kps, features)      # 返回特征点集，及对应的描述特征

    ##################################################################################
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.BFMatcher()                                   # 建立暴力匹配器
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)      # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2

        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:   # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                matches.append((m[0].trainIdx, m[0].queryIdx))          # 存储两个点在featuresA, featuresB中的索引值

        if len(matches) > 4:        # 当筛选后的匹配对大于4时，计算视角变换矩阵
            # 投影变换矩阵：3*3。有八个参数对应八个方程，其中一个为1用于归一化。对应四对，每对(x, y)
            ptsA = np.float32([kpsA[i] for (_, i) in matches])          # 获取匹配对的点坐标
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)      # 使用RANSAC算法利用匹配特征向量估计单映矩阵（homography：单应性）
            return (matches, H, status)

        return None     # 如果匹配对小于4时，返回None

    ##################################################################################
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA        # 将A、B图左右连接到一起
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:      # 当点对匹配成功时，画到可视化图上
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis      # 返回可视化结果


##################################################################################
if __name__ == '__main__':
    # 读取拼接图片
    imageA = cv2.imread("left_01.png")
    imageB = cv2.imread("right_01.png")

    # 把图片拼接成全景图
    stitcher = Stitcher()       # 调用拼接函数
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

    # 显示所有图片
    cv2.imshow("Image A", imageA)
    cv2.imshow("Image B", imageB)
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

