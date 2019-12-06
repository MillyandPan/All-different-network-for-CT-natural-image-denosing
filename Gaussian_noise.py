# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 12:50:55 2019

@author: span21
Add Guassian Noise
"""

import cv2
import math
import numpy as np
import os

base = './image/val_256'
output1 = './image/val_256_Gaussian_noise1'
output2 = './image/val_256_Gaussian_noise2'
output3 = './image/val_256_Gaussian_noise3'

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

Gaussian1 = AddNoise()
Gaussian1.Gaussian(0,20,output1)
Gaussian1.Gaussian(0,50,output2)
Gaussian1.Gaussian(0,100,output3)
