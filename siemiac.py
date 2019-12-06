# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:04:51 2019

@author: span21
"""

import torchvision
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import os
import cv2
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import scipy.io

#The settings
training_dir = './quicktest_cnn/DL_project_data'
testing_dir = "./data/faces/testing/"
train_batch_size = 100
train_number_epochs = 100
image_height = 64
image_width = 64
piece_height = 32
piece_width = 32

#Function for reading images
def read_image(path,input_height,input_width):
    _img_ = cv2.imread(path)
    img = np.ones([input_height, input_width,3])
    for channel in range(3):
        img[:,:,channel] = cv2.resize(_img_[:,:,channel], dsize=(input_height, input_width), interpolation=cv2.INTER_CUBIC)
    return img.astype(np.float32)


#Function to convert the images to tensors and create the labels
#Each image is divided into 4 pieces: lefttop:0, righttop:1, leftbottom:2, rightbottom:3
class InverseSiameseNetworkDataset():
    def __init__(self,dataset_dir,transform=None):
        self.dataset_dir = dataset_dir    
        self.transform = transform
        self.Files = os.listdir(dataset_dir+'/image')
        
    def __getitem__(self,index):
        
        #The original image, is the label for reordering.
        Files = os.listdir(self.dataset_dir+'/label')
        reorder_labels = read_image(self.dataset_dir+'/label'+ '/' +Files[index],image_height,image_width)
        reorder_labels = self.transform(reorder_labels)
        
        #Read each corresponding image pieces and shuffle them then make them together
        images = os.listdir(self.dataset_dir+'/image')
        images_folder = images[index]
        image_list = os.listdir(self.dataset_dir+'/image/'+images_folder)        
        random.shuffle(image_list)
        image = np.ones([image_height,image_width,3])
        x_start = 0
        y_start = 0
        for ind,label in enumerate(image_list):
            piece = read_image(self.dataset_dir+'/image'+ '/' +images_folder+'/'+image_list[ind],piece_height,piece_width)
            image[int(y_start):int(y_start+piece_height),int(x_start):int(x_start+piece_width),:] = piece
            x_start += piece_width
            if x_start>=image_width:
                x_start = 0
                y_start += piece_height
        image = self.transform(image)
        
        #Read each piece and make the pairwise label as the next index of the image. For example, if the piece is shuffled as [2,1,3,0], the pairwise label should be 
        #[3,2,0,1]
        pair_labels = np.zeros([1,int((image_height/piece_height)*(image_width/piece_width))])
        for ind,label in enumerate(image_list):
            name = image_list[ind].split(".")[0]
            pair_labels[0,ind] = int((float(name)+1)%((image_height/piece_height)*(image_width/piece_width)))
        pair_labels = self.transform(pair_labels)
        
        sample = {'image': image, 'reorder_labels': reorder_labels, 'pair_labels': pair_labels}
        return sample

    
    def __len__(self):
        return len(self.Files)

siamese_dataset = InverseSiameseNetworkDataset(dataset_dir=training_dir,transform=transforms.Compose([transforms.ToTensor()]))

#The model
class InverseSiameseNetwork(nn.Module):
    def __init__(self,kernel_size = 3,stride = 1,padding =1, bias = True):
        super(InverseSiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=kernel_size,stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.MaxPool2d(2),
            
            nn.Conv2d(9, 27, kernel_size=kernel_size,stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.MaxPool2d(2),
                        
            nn.Conv2d(27, 54, kernel_size=kernel_size,stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.BatchNorm2d(54),
            nn.MaxPool2d(2)
        )

        self.reorder_cnn = nn.Sequential(         
            nn.ConvTranspose2d(54, 27, kernel_size=4,stride = 2, padding = padding, bias = True),
            nn.ReLU(),
            nn.BatchNorm2d(27),
            
            nn.ConvTranspose2d(27, 9, kernel_size=4,stride = 2, padding = padding, bias = True),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            
            nn.ConvTranspose2d(9, 3, kernel_size=4,stride = 2, padding = padding, bias = True),
            nn.ReLU(),
            )
        
        self.pair_cnn = nn.Sequential(            
            nn.Conv2d(54, 27, kernel_size=kernel_size,stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.BatchNorm2d(27),
            nn.MaxPool2d(2),
            nn.Conv2d(27, 9, kernel_size=kernel_size,stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 1, kernel_size=kernel_size,stride = stride, padding = padding, bias = bias),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            )
        
#        self.fully_connected = nn.Sequential(            
#            nn.Linear(8*3*8,24),
#            nn.ReLU(),
#            nn.Linear(24, 8),
#            nn.ReLU(),
#            nn.Linear(8, 4),
#            )
                

    def forward(self, input_image):
        reorder_output = self.reorder_cnn(self.cnn1(input_image))
        pair_output = self.pair_cnn(self.cnn1(input_image))
##        pair_output = pair_output.argmax(2)
#        pair_output = pair_output.view(-1,1*4*4)
#        pair_output = self.fully_connected(pair_output)
        return reorder_output, pair_output

##This function is a speical cross_entropy for a seq label: our pairwise label is a seq instead of one single value so we need this
#def special_cross_entropy(prediction,label):
#    #each column is one index for the pair of the piece
#    loss = torch.zeros([1,1])
#    prediction = prediction.squeeze()
#    label = prediction.squeeze()
#    for piece in range(prediction.size()[1]):
#        loss += torch.nn.functional.cross_entropy(prediction[:,piece,:],label[:,piece,:])
#    return loss

#The total loss function: I don't have good idea to change the weight
class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
    def forward(self, output1,output2,pair_recon_output,reorder_label,pair_label):
        self_loss = nn.functional.mse_loss(output1, pair_recon_output)
        reorder_loss = nn.functional.mse_loss(output1, reorder_label)
        
        output2 = output2.squeeze()
        pair_label = pair_label.squeeze()
        pair_loss = torch.nn.functional.cross_entropy(output2, pair_label.long())
        the_loss = self_loss+reorder_loss+pair_loss*1000
#        print('self_loss'+str(self_loss))
#        print('reorder_loss'+str(reorder_loss))
#        print('pair_loss'+str(pair_loss))
        return the_loss

#This function is to use the pairwise index to reconstruct the image. Since the pairwise index is the "next piece" so I minus 1
def pairwise_recon(index, image_array):
    sepe_ratio = image_height/piece_height
    img = torch.ones([index.size(0),3,image_height,image_width])
    for batch in range(index.size(0)):
        for piece_idx in range(int(sepe_ratio**2)):
            pair_ind = index[batch,:,:,piece_idx].argmax()
            x_ind = ((pair_ind+sepe_ratio**2-1)%(sepe_ratio**2))%sepe_ratio
            y_ind = np.floor((pair_ind+sepe_ratio**2-1)%(sepe_ratio**2)/sepe_ratio)
            
            recon_x_ind = piece_idx%sepe_ratio
            recon_y_ind = np.floor(piece_idx/sepe_ratio)
            
            img[batch,:,int(y_ind*piece_height):int((y_ind+1)*piece_height),int(x_ind*piece_width):int((x_ind+1)*piece_width)] =\
            image_array[batch,:,int(recon_y_ind*piece_height):int((recon_y_ind+1)*piece_height),int(recon_x_ind*piece_width):int((recon_x_ind+1)*piece_width)]
    return img.float()


    
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=False,
                        batch_size=train_batch_size)    

test_dataloader = DataLoader(siamese_dataset,
                        shuffle=False,
                        batch_size=train_batch_size)  

net = InverseSiameseNetwork().cuda()
criterion = Loss()
optimizer = optim.Adam(net.parameters(),lr = 0.01)
for epoch in range(0,train_number_epochs):
    total_loss = 0
    for i, data in enumerate(train_dataloader,0):
        img = data['image'].float().cuda() 
        reorder_label = data['reorder_labels'].float().cuda() 
        pair_label = data['pair_labels'].float().cuda() 
        optimizer.zero_grad()
        output1,output2 = net(img)
        pair_recon_output = pairwise_recon(output2.cpu(),img.cpu())
        pair_recon_output = pair_recon_output.cuda()
        loss = criterion(output1,output2.float(),pair_recon_output,reorder_label,pair_label)
        total_loss+=loss
        loss.backward()
        optimizer.step()
    print('Epoch: '+ str(epoch) + ' has loss:' + str(total_loss))
    data = dict(loss = total_loss, epoch = epoch)
    scipy.io.savemat(os.path.join('./quicktest_cnn/train_output_epoch_{0:06d}'.format(epoch)), data)
    torch.save(net, './quicktest_cnn/god.path')