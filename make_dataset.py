import os
import cv2
import numpy as np
import random

input_height = 256  # 768
input_width = 256  # 1024


# Limit the size of the message to be within the max_size
def resize_content_img(img):
    return cv2.resize(img, dsize=(input_height, input_width), interpolation=cv2.INTER_AREA)


def read_image(path):
    _img_ = cv2.imread(path)
    return _img_.astype(np.float32)


def load_content_img(path):
    _img_ = read_image(path)
    _img_ = resize_content_img(_img_)
    return _img_

def main():
    base = './image/'
    image_folder = 'val_256_Gaussian_noise3'
    label_folder = 'val_256'

    train_files = []
    train_labels = []

    eval_files = []
    eval_labels = []

    vali_files = []
    vali_labels = []

    image_files = os.listdir(base + image_folder + '/')
    random.shuffle(image_files)
    train_set_num = int(6400)
    eval_set_num = int(6400+1600)
    train_set = image_files[:train_set_num]
    eval_set = image_files[train_set_num:eval_set_num]
    vali_set = image_files[eval_set_num:]
    for file in train_set:
        train_files.append(base + image_folder + '/' + file)
        train_labels.append(base + label_folder + '/' + file)
    for file in eval_set:
        eval_files.append(base + image_folder + '/' + file)
        eval_labels.append(base + label_folder + '/' + file)
    for file in vali_set:
        vali_files.append(base + image_folder + '/' + file)
        vali_labels.append(base + label_folder + '/' + file)
    np.save('train_imgs', train_files)
    np.save('train_lbs', train_labels)
    np.save('eval_imgs', eval_files)
    np.save('eval_lbs', eval_labels)
    np.save('valid_imgs', vali_files)
    np.save('valid_lbs', vali_labels)



if __name__ == '__main__':
    print('starting making dataset...')
    main()
    print('finished making dataset')
