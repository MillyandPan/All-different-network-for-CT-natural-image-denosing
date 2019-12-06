# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:01:51 2019

@author: span21
"""
import os
import cv2
import numpy as np
import random

input_height = 256  # 768
input_width = 256  # 1024

def read_image(path):
    _img_ = cv2.imread(path)
    return _img_.astype(np.float32)


def main():
    base = './image/'
    image_folder = 'val_256_Gaussian_noise3/'
    label_folder = 'val_256/'
    aa = cv2.imread(base+image_folder+'Places365_val_00000031.jpg');
    a = 1
    count = 0
    img_files = np.load('train_lbs.npy')
#    for ind in range(len(img_files)):
#        bb = cv2.imread(img_files[ind])
#        hei,wid,ch = bb.shape
#        if ch != 3 | hei !=256 | wid != 256:
#            count += 1
    print(str(count))
    print(str(len(img_files)))
if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')