# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:23:03 2019

@author: span21
Calculate MSE
"""

import cv2
import numpy as np

class MSE():
    global length
    global width
    global depth

    def __init__(self,length,width,depth):
        self.length = length
        self.width = width
        self.depth = depth
        
    def calculate_MSE_3d(self,imagex,imagey):
        thesum = 0.0
        resized_x = np.zeros([self.length,self.width,self.depth])
        resized_y = np.zeros([self.length,self.width,self.depth])
        resized_x[:,:,0] = cv2.resize(imagex[:,:,0],(self.length,self.width)) 
        resized_x[:,:,1] = cv2.resize(imagex[:,:,1],(self.length,self.width)) 
        resized_x[:,:,2] = cv2.resize(imagex[:,:,2],(self.length,self.width)) 
        resized_y[:,:,0] = cv2.resize(imagey[:,:,0],(self.length,self.width)) 
        resized_y[:,:,1] = cv2.resize(imagey[:,:,1],(self.length,self.width)) 
        resized_y[:,:,2] = cv2.resize(imagey[:,:,2],(self.length,self.width)) 
        for x in range(1,self.length):
            for y in range(1,self.width):
                for z in range(1,self.depth):
                    difference = (resized_x[x,y,z]-resized_y[x,y,z])**2
                    thesum += difference
        RMSE = thesum/(self.length*self.width*self.depth)
        return RMSE
    
aa = MSE(256,256,3)
image1 =cv2.imread('./image/val_256/Places365_val_00009662.jpg')
image2 = cv2.imread('./image/val_256_Gaussian_noise3/Places365_val_00009662.jpg')
MSE = aa.calculate_MSE_3d(image1,image2)
    
        
        
