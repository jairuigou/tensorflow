import cv2
import os
import random
import numpy as np
PATH = "/mnt/c/captchas/test_rgb"
os.chdir(PATH)
jpgs = os.listdir()
dic = {'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}

for jpg in jpgs:
    img = cv2.imread(jpg)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.cv2.threshold(grayimg,200,255,0)
    print(jpg)
    cv2.imwrite("../test_binary/"+jpg,thresh)
    cut = []
    cut.append(thresh[2:22,2:16])
    cut.append(thresh[2:22,16:30])
    cut.append(thresh[2:22,30:44])
    cut.append(thresh[2:22,44:58])
    for j in range(4):
        savepath = "../test_img/" + jpg[j] + "/" + str(dic[jpg[j]])+".jpg"
        cv2.imwrite(savepath,cut[j])
        dic[jpg[j]] += 1