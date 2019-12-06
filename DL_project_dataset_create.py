# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:24:26 2019

@author: span21
"""

import cv2
import numpy as np
import os

base = './DL_project_data/label'
output1 = './DL_project_data/image'
input_height = 64
input_width = 64
sepe_ratio = 2

def read_image(path):
    _img_ = cv2.imread(path)
    img = np.ones([input_height, input_width,3])
    for channel in range(3):
        img[:,:,channel] = cv2.resize(_img_[:,:,channel], dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32)

class sepe():
    
    def __init__(self):
        self = self
        
    def sepe(self,output):
        Files = os.listdir(base)
        for file in Files:
            image = read_image(base+ '/' +file)
            if not os.path.isdir(output+'/'+file.split('.')[0]):
                os.mkdir(output+'/'+file.split('.')[0])
            x_start = 0
            y_start = 0
            for sepe_idx in range(sepe_ratio**2):
                img = image[int(y_start):int(y_start+input_height/sepe_ratio),int(x_start):int(x_start+input_width/sepe_ratio),:]
                x_start += input_width/sepe_ratio
                if x_start>=input_width:
                    x_start = 0
                    y_start += input_height/sepe_ratio
                cv2.imwrite(output+'/'+file.split('.')[0]+'/'+str(sepe_idx)+'.jpg', img)
Sepe = sepe()
Sepe.sepe(output1)    