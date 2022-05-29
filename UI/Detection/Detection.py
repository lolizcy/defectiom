import sys
import time
import os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QIcon, QPixmap, QStandardItem
from UI.Detection.DetectionWin import DetectionWin
from UI.Setting.Setting import Setting
from utils.CommonHelper import CommonHelper
from Service.hkDetect import hkDetect
import numpy as np
import cv2
from model.Opt import Opt
from model.Camera import Camera
from numba import jit

from  MvCameraControl_class import *
winfun_ctype = WINFUNCTYPE
DETECTION_THRESHOLD = 4000
OUT_PATH = "../../output/Image/"

class Detection(DetectionWin):
    deviceList = MV_CC_DEVICE_INFO_LIST()
    cam = MvCamera()
    def __init__(self,configuration):
        super(Detection,self).__init__()
        self.setupUi(self)
        styleFile = '../../resource/Detection.qss'
        # 换肤时进行全局修改，只需要修改不同的QSS文件即可
        style = CommonHelper.readQss(styleFile)
        self.setStyleSheet(style)
        self.cams={}
        self.configuration = configuration
        self.g_bExit = False
        self.cameraDetectionLabels = {"1":self.camera1_detection_label,"2":self.camera2_detection_label,"3":self.camera3_detection_label,"4":self.camera4_detection_label}
        self.cameraResultLabels = {"1":self.camera1_result_label,"2":self.camera2_result_label,"3":self.camera3_result_label,"4":self.camera4_result_label}
        self.GreenButton.clicked.connect(self.detect)
        self.RedButton.clicked.connect(self.stop)
        self.BlueButton.clicked.connect(self.reset)

    def detect(self):
        self.opt = Opt()
        self.opt.cfg = self.configuration.value("CFG_PATH")
        self.opt.output = self.configuration.value("SAVE_IMG_PATH")
        self.opt.weights = self.configuration.value("WEIGHTS_PATH")
        # self.hkDetect = hkDetect(self.opt)
        for camNo in self.configuration.value('CAM_LIST'):
            cam = Camera(camNo,self.opt,0)
            cam.show_picture_signal.connect(self.image_show)
            cam.detect_show_signal.connect(self.origin_image_show)
            self.cams.update({camNo: cam})
            cam.openCam()




    def stop(self):
        for key in self.cams:
            self.cams[key].g_bExit = True
            self.cams[key].closeCam()

    def reset(self):
        self.speedValue = 0.0
        self.ratio = 0.0
        self.detect_num = 0
        self.good_num = 0
        self.hkDetect = None
        self.detection_number_value.setText(str(self.detect_num) + "个")
        self.speed_value.setText(str(self.speed) + "个/分钟")
        self.ratio_value.setText(str(self.ratio) + "%")

    @jit
    def select_side(self,img):
        (h, w) = img.shape
        imagex = img

        # 初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
        a = [0 for z in range(0, w)]

        for i in range(0, w):  # 遍历每一列
            for j in range(0, h):  # 遍历每一行
                if imagex[j, i] == 0:  # 判断该点是否为黑点，0代表是黑点
                    a[i] += 1  # 该列的计数器加1
                    imagex[j, i] = 255  # 记录完后将其变为白色，即等于255
                if a[i] >= 0.4 * h:
                    a[i] = h

        for i in range(0, w):  # 遍历每一列
            for j in range(h - a[i], h):  # 从该列应该变黑的最顶部的开始向最底部设为黑点
                imagex[j, i] = 0  # 设为黑点
        # hh = int(0.4 * h)
        # for i in range(0, w):  # 遍历每一列
        #     for j in range(0, hh):  # 遍历每一行
        #         if imagex[j, i] == 0:  # 判断该点是否为黑点，0代表是黑点
        #             a[i] += 1  # 该列的计数器加1
        #             imagex[j, i] = 255  # 记录完后将其变为白色，即等于255
        #         if a[i] >= 0.16 * h:
        #             a[i] = h
        #
        # for i in range(0, w):  # 遍历每一列
        #     for j in range(h - a[i], h):  # 从该列应该变黑的最顶部的开始向最底部设为黑点
        #         imagex[j, i] = 0  # 设为黑点
        return imagex

    def draw(self,img):
        src_image = img
        # cv2.imshow("src_image", src_image)


        gauss_image = cv2.GaussianBlur(src_image, (5, 5), 0)
        gauss_image = cv2.GaussianBlur(gauss_image, (5, 5), 0)
        # cv2.imshow("Gauss", gauss_image)
        # 求图像像素点均值


        means, dev = cv2.meanStdDev(gauss_image)
        # 确定二值化阈值
        cut = int(means) * 0.65
        # cut = int(means) - 1.6 * int(dev)


        ret, binary = cv2.threshold(src_image, cut, 255, cv2.THRESH_BINARY_INV)
        # img = cv2.threshold(src_image,cv2.THRESH_BINARY,135)
        # ret, binary = cv2.threshold(src_image, 80, 255, cv2.THRESH_BINARY)
        # ret, binary = cv2.threshold(src_image, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

        # # cv2.imshow("THRESH_BINARY", binary)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dst = cv2.erode(binary, kernel)  # 腐蚀操作
        # # cv2.imshow("erode_demo", dst)
        # dst1 = cv2.dilate(dst, kernel)
        # # cv2.imshow("dilate_demo", dst1)
        # #找轮廓
        # contours, hierarchy = cv2.findContours(dst1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        dst = cv2.erode(binary, kernel5)# 腐蚀操作
        dst = cv2.dilate(dst, kernel5)

        imagex = self.select_side(binary)

        # (h, w) = binary.shape
        # imagex = binary
        #
        # # 初始化一个跟图像宽一样长度的数组，用于记录每一列的黑点个数
        # a = [0 for z in range(0, w)]
        #
        # for i in range(0, w):  # 遍历每一列
        #     for j in range(0, h):  # 遍历每一行
        #         if imagex[j, i] == 0:  # 判断该点是否为黑点，0代表是黑点
        #             a[i] += 1  # 该列的计数器加1
        #             imagex[j, i] = 255  # 记录完后将其变为白色，即等于255
        #         if a[i] >= 0.4 * h:
        #             a[i] = h
        #
        # for i in range(0, w):  # 遍历每一列
        #     for j in range(h - a[i], h):  # 从该列应该变黑的最顶部的开始向最底部设为黑点
        #         imagex[j, i] = 0  # 设为黑点

        imagex = cv2.dilate(imagex, kernel5)

        dst1 = cv2.subtract(dst, imagex)
        dst1 = cv2.erode(dst1, kernel5)
        dst1 = cv2.dilate(dst1, kernel5)
        # cv2.imshow("黑白图片",dst)
        # cv2.imshow("2",imagex)
        # cv2.imshow("差值",dst1)
        # cv2.waitKey(0)

        contours, hierarchy = cv2.findContours(dst1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            # print("轮廓 %d 的面积是:%d" % (i, area))
            if area >= 200:
                # 确定最小外接矩形，并框出
                rect = cv2.minAreaRect(contours[i])
                if rect[1][1] > 14 and rect[1][0] > 14:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    src_image = cv2.UMat(src_image).get()
                    cv2.drawContours(src_image, [box], 0, (255, 255, 111), 3)

        return src_image


    def origin_image_show(self,image,camNo):
        camera = self.cams[camNo]

        image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image2 = image1
        image2 = cv2.GaussianBlur(image2, (9, 9), 0)
        grad_x = cv2.Sobel(image2, cv2.CV_32F, 1, 0, 1)
        gradx = cv2.convertScaleAbs(grad_x)
        ret, binary = cv2.threshold(gradx, 60, 255, cv2.THRESH_BINARY)#二值化

        a, b = image2.shape

        print(a, b)
        # for i in range(b):
        #     if binary[1, i] == 255:
        #         # print(image1[1, i])
        #         image2[1, i] = 255
        #         # print(i)
        #         x = i
        #
        #         break
        #
        # # print(l1)
        # # print("--------------------------")
        # for j in range(b - 1, -1, -1):
        #     if binary[a - 1, j] == 255:
        #         # print(image1[a - 1, j])
        #         # image2[a - 1, j - 1] = 255
        #         # image2[a - 1, j + 1] = 255
        #         y = j
        #         # print(j)
        #
        #         break
        #
        # #确定每个磁芯的边间，并切割
        # cropped0 = image1[0:a, x:int(x) + int((y - x) / 10)]
        # cropped1 = image1[0:a, int(x) + int((y - x) / 10):int(x) + int(2 * (y - x) / 10)]
        # cropped2 = image1[0:a, int(x) + int(2 * (y - x) / 10):int(x) + int(3 * (y - x) / 10)]
        # cropped3 = image1[0:a, int(x) + int(3 * (y - x) / 10):int(x) + int(4 * (y - x) / 10)]
        # cropped4 = image1[0:a, int(x) + int(4 * (y - x) / 10):int(x) + int(5 * (y - x) / 10)]
        # cropped5 = image1[0:a, int(x) + int(5 * (y - x) / 10):int(x) + int(6 * (y - x) / 10)]
        # cropped6 = image1[0:a, int(x) + int(6 * (y - x) / 10):int(x) + int(7 * (y - x) / 10)]
        # cropped7 = image1[0:a, int(x) + int(7 * (y - x) / 10):int(x) + int(8 * (y - x) / 10)]
        # cropped8 = image1[0:a, int(x) + int(8 * (y - x) / 10):int(x) + int(9 * (y - x) / 10)]
        # cropped9 = image1[0:a, int(x) + int(9 * (y - x) / 10):int(x) + int(y - x)]

        first_line = 130
        wid = 129
        cropped0 = image1[0:a, first_line:wid]
        cropped1 = image1[0:a, first_line+wid:first_line+wid*2]
        cropped2 = image1[0:a, first_line+wid*2:first_line+wid*3]
        cropped3 = image1[0:a, first_line+wid*3:first_line+wid*4]
        cropped4 = image1[0:a, first_line+wid*4:first_line+wid*5]
        cropped5 = image1[0:a, first_line+wid*5:first_line+wid*6]
        cropped6 = image1[0:a, first_line+wid*6:first_line+wid*7]
        cropped7 = image1[0:a, first_line+wid*7:first_line+wid*8]
        cropped8 = image1[0:a, first_line+wid*8:first_line+wid*9]
        cropped9 = image1[0:a, first_line+wid*9:first_line+wid*10]

        #进行磁芯缺陷检测
        cropped0 = self.draw(cropped0)
        cropped1 = self.draw(cropped1)
        cropped2 = self.draw(cropped2)
        cropped3 = self.draw(cropped3)
        cropped4 = self.draw(cropped4)
        cropped5 = self.draw(cropped5)
        cropped6 = self.draw(cropped6)
        cropped7 = self.draw(cropped7)
        cropped8 = self.draw(cropped8)
        cropped9 = self.draw(cropped9)
        #合并图像
        lst5 = [cropped0, cropped1, cropped2, cropped3, cropped4, cropped5, cropped6, cropped7, cropped8, cropped9]

        fin_image = np.hstack(lst5)
        lst5 = []

        # image = cv2.resize(image, (self.opt.width, self.opt.height))
        image = cv2.resize(fin_image, (self.opt.width, self.opt.height))
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


        result_image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                    QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式

        self.cameraDetectionLabels[camNo].setPixmap(QtGui.QPixmap.fromImage(result_image))

    def image_show(self,image,defect_type,camNo):
        camera = self.cams[camNo]
        image = cv2.resize(image, (self.opt.width, self.opt.height))
        # originalshow = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # area = self.hkDetect.ostu(originalshow)
        # if(area>DETECTION_THRESHOLD):
        #     camera.imgs.append(originalshow)
        # elif(len(camera.imgs)):
        #     camera.detect_num+=1
        #     select_img = camera.imgs[int(len(camera.imgs)/2)]
        #     result_image,defect_classes = self.hkDetect.detect(select_img, self.opt)
        #
        #     if(len(defect_classes)==1 or "good" in defect_classes ):
        #         camera.good_num+=1
        #         self.cameraResultLabels[camNo].setStyleSheet("color:green;font-size:14px")
        #     else:
        #          camera.bad_num+=1
        #          self.cameraResultLabels[camNo].setStyleSheet("color:red;font-size:14px")
        result_image = QtGui.QImage(image.data, image.shape[1], image.shape[0],
                                QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.cameraDetectionLabels[camNo].setPixmap(QtGui.QPixmap.fromImage(result_image))
        defect_type = ''.join(defect_type)
        self.cameraResultLabels[camNo].setText(defect_type)
        row = int(camNo)-1
        self.model.setItem(row,1, QStandardItem(str(camera.detect_num)))
        self.model.setItem(row,2, QStandardItem(str(camera.good_num)))
        self.model.setItem(row,3, QStandardItem(str(camera.bad_num)))
        self.model.setItem(row,4, QStandardItem(str((camera.bad_num/camera.detect_num)*100)+"%"))
        camera.imgs.clear()

    #
    # def work_thread(self, cam=0, pData=0, nDataSize=0, camNo=0):
    #     stFrameInfo = MV_FRAME_OUT_INFO_EX()
    #     memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    #     while True:
    #         ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
    #         if ret == 0:
    #
    #             print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
    #                 stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
    #             image = np.asarray(pData).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    #             self.image_show(image,camNo)
    #         else:
    #             print("no data[0x%x]" % ret)
    #         if self.cams[camNo].g_bExit == True:
    #             break


