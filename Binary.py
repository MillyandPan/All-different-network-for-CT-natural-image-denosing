# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:22:44 2019

@author: span21
"""

import cv2
import math
import numpy as np
import os

base = './image/val_256'
output1 = './image/val_256_binary'

def read_image(path):
    _img_ = cv2.imread(path)
    return _img_.astype(np.float32)

class AddNoise():
    
    def __init__(self):
        self = self
        
    def Gaussian(self,mean,variance,output):
        Files = os.listdir(base)
        for file in Files:
            image = read_image(base+ '/' +file)
            row,col,ch= image.shape
            gauss = np.random.normal(mean,np.sqrt(variance),(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            if not os.path.isdir(output):
                os.mkdir(output)
            if os.path.exists(output+'/'+file):
                os.remove(output+'/'+file)
            cv2.imwrite(output+'/'+file, noisy)