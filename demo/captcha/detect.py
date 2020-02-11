import tensorflow as tf
import cv2
import os
import numpy as np
model = tf.keras.models.load_model('captcha_model')
PATH = '/home/jairui/work/cc/'
os.chdir(PATH)
jpgs = os.listdir()

for jpg in jpgs:
    img = cv2.imread(jpg)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.cv2.threshold(grayimg,200,255,0)
    num = np.zeros((4,20,14))
    num[0] = thresh[2:22,2:16]
    num[1] = thresh[2:22,16:30]
    num[2] = thresh[2:22,30:44]
    num[3] = thresh[2:22,44:58]
    num = num.reshape(4,20,14,1)
    res = model.predict_classes(num)
    resstr = ''
    for i in range(4):
        resstr += str(res[i])
    os.rename(jpg,resstr+'.jpg')