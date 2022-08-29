import cv2
import sys
import math
import numpy as np
import pytesseract

class Captcha(object):
    def __init__(self):
        pass
    def __call__(self,im_path,save_path):
        query_img=cv2.imread(im_path,cv2.IMREAD_GRAYSCALE)
        query_img=cv2.resize(query_img,None,fx=9,fy=9)
        query_img=self.filter_img_pixel(query_img)
        recognised_characters=self.recognise(query_img)
        np.savetxt(save_path,[recognised_characters],fmt='%s')
        print('output from character recognition: ',recognised_characters)
        pass
    def filter_img_pixel(self,img):
        img[img>128]=255
        img[img<=128]=0
        return img
    def recognise(self,img):
        return pytesseract.image_to_string(img)

if __name__=='__main__':
    captcha=Captcha()
    captcha(sys.argv[1],sys.argv[2])

