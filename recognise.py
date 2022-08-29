import os
import cv2
import sys
import numpy as np

class Captcha(object):
    def __init__(self):
        self.img_template_hog_vectors,self.characters=self.preprocessing()
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
    def preprocessing(self):
        img_template_hog_vectors=[]
        characters=[]
        for i in range(0,25):
            output_name='output/output'+format(i,'02d')+'.txt'
            if not os.path.exists(output_name):
                continue
            output_characters=str(np.loadtxt(output_name,dtype='str'))
            input_name='input/input'+format(i,'02d')+'.jpg'
            img=cv2.imread(input_name,cv2.IMREAD_GRAYSCALE)
            img=cv2.resize(img,None,fx=9,fy=9)
            img=self.filter_img_pixel(img)
            character_regions=self.extract_characters(img)
            if len(character_regions)!=len(output_characters):
                continue
            for j in range(len(character_regions)):
                character_box=character_regions[j]
                x,y,w,h=character_box
                character_img=img[y:y+h,x:x+w]
                img_template_hog_vector=self.extract_character_hog_vector(character_img)
                img_template_hog_vectors.append(img_template_hog_vector)
                characters.append(output_characters[j])
        img_template_hog_vectors=np.array(img_template_hog_vectors)
        characters=np.array(characters)
        return img_template_hog_vectors,characters
    def extract_characters(self,img):
        img=cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_REPLICATE)
        img=cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        contours,hierarchy=cv2.findContours(img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        character_regions=[]
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            character_regions.append((x,y,w,h))
        character_regions=sorted(character_regions,key=lambda x:x[0])
        return character_regions
    def extract_character_hog_vector(self,character_img):
        character_img=cv2.copyMakeBorder(character_img,20,20,20,20,cv2.BORDER_REPLICATE)
        character_img=cv2.resize(character_img,(64,128))
        img_template_hog_vector=cv2.HOGDescriptor().compute(character_img)
        return img_template_hog_vector
    def recognise(self,query_img):
        character_regions=self.extract_characters(query_img)
        recognised_character=''
        for j in range(len(character_regions)):
            character_box=character_regions[j]
            x,y,w,h=character_box
            character_img=query_img[y:y+h,x:x+w]
            query_vector=self.extract_character_hog_vector(character_img)
            dist=None
            current_char=None
            for k in range(len(self.img_template_hog_vectors)):
                img_template_hog_vector=self.img_template_hog_vectors[k]
                current_dist=np.sum((query_vector-img_template_hog_vector)**2)
                if dist==None or current_dist<dist:
                    dist=current_dist
                    current_char=self.characters[k]
            recognised_character=recognised_character+current_char
        return recognised_character

if __name__=='__main__':
    captcha=Captcha()
    captcha(sys.argv[1],sys.argv[2])

