# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:22:44 2019

@author: span21
"""

import cv2
import numpy as np
import os

base = './image/val_256'
output1 = './image/val_256_binary'

def read_image(path):
    _img_ = cv2.imread(path)
    return _img_.astype(np.float32)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

class Gray():
    
    def __init__(self):
        self = self
        
    def gray(self,output):
        Files = os.listdir(base)
        for file in Files:
            image = read_image(base+ '/' +file)
            Gray = rgb2gray(image)
            if not os.path.isdir(output):
                os.mkdir(output)
            if os.path.exists(output+'/'+file):
                os.remove(output+'/'+file)
            cv2.imwrite(output+'/'+file, Gray)

Gray1 = Gray()
Gray1.gray(output1)    