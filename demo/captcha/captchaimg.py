import cv2
import os
import numpy as np

TRAIN_PATH = "/mnt/c/captchas/train_img/"
TEST_PATH = "/mnt/c/captchas/test_img/"
TRAIN_NPY_PATH = {'image':TRAIN_PATH+'train_img.npy','labels':TRAIN_PATH+'train_labels.npy'}
TEST_NPY_PATH = {'image':TEST_PATH+'test_img.npy','labels':TEST_PATH+'test_labels.npy'}

class CaptchaImg:
    def init_array(self,path):
        img_num = 0
        shape0 = -1
        shape1 = -1
        for i in range(10):
            subpath = path+str(i)+'/'
            if os.path.exists(subpath):
                img_num += len(os.listdir(subpath))
        for i in range(10):
            subpath = path+str(i)+'/'
            if os.path.exists(subpath):
                jpgs = os.listdir(subpath)
                jpg = cv2.imread(subpath + jpgs[0])
                shape0 = jpg.shape[0]
                shape1 = jpg.shape[1]
                break
        if shape0==-1 or shape1==-1:
            shape0 = 20
            shape1 = 20
        if img_num == 0:
            img_num = 1000
        return np.zeros((img_num,shape0,shape1)),np.zeros((img_num))
    def load_rawimg_data(self,path):
        os.chdir(path)
        nparr_img,nparr_labels = self.init_array(path)
        index = 0
        for i in range(10):
            subpath = path+str(i)+'/'
            if os.path.exists(subpath):
                os.chdir(subpath)
                jpgs = os.listdir()
                print(subpath)
                for jpg in jpgs:
                    img = cv2.imread(jpg)
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    nparr_img[index] = gray
                    nparr_labels[index] = i
                    index += 1
        return nparr_img,nparr_labels 
    def load_data(self,path,npypath):
        npy_imgpath = npypath['image']
        npy_labelspath = npypath['labels']
        if not os.path.exists(npy_imgpath) or not os.path.exists(npy_labelspath):
            np_img,np_labels = self.load_rawimg_data(path)
            np.save(npy_imgpath,np_img)
            np.save(npy_labelspath,np_labels)
        return np.load(npy_imgpath),np.load(npy_labelspath)
    def load_train_data(self):
        return self.load_data(TRAIN_PATH,TRAIN_NPY_PATH)
    def load_test_data(self):
        return self.load_data(TEST_PATH,TEST_NPY_PATH)
        