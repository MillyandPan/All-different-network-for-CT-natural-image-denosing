# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:25:51 2019

@author: pan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:23:29 2019

@author: pan
"""

## Some general imports we may need:
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import time
import struct
import torch.optim as optim
import torchvision
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
import time
import torchvision.models as models
from torch.autograd import Variable

gpu_boole = torch.cuda.is_available()
print(gpu_boole)
batchsize = 100
#number of epochs to train for:
epochs = 100

## Defining the model:
class Net(nn.Module):
  def __init__(self, dim_in, kernel_size=3, stride=1, padding=1, bias=True):
    super(Net, self).__init__()
   
    ##feedfoward layers:
    self.conv1 = nn.Conv2d(dim_in, 8, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 32, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    self.bn3 = nn.BatchNorm2d(64)
    self.fc1 = nn.Linear(64 * 28 * 28, 64)
    self.fc2 = nn.Linear(64, 10)

    ##activations:
    self.relu = nn.LeakyReLU(0.1)
               
  def forward(self, input_data):
    out = self.relu(self.bn1(self.conv1(input_data)))
    out = self.relu(self.bn2(self.conv2(out)))
    out = self.relu(self.bn3(self.conv3(out)))
    out = out.view(batchsize,-1)
    out = self.fc2(self.fc1(out))
    return out #returns class probabilities for each image

net = Net(dim_in = 1)
if gpu_boole:
  net = net.cuda()

patch  = torch.ones((1,28,28))
optimizer = torch.optim.SGD(net.parameters(), lr = 0.01)
optimizer_adv = torch.optim.Adam([patch.requires_grad_()], lr = 0.0001)
loss_metric = nn.CrossEntropyLoss()
MSE_loss = torch.nn.MSELoss()
pytorch_transforms = torchvision.transforms.Compose([transforms.ToTensor()])
   
train_dataset = FashionMNIST(root ='./data', transform=pytorch_transforms, train=True, download=True)
test_dataset = FashionMNIST(root ='./data', transform=pytorch_transforms, train=False, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batchsize, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = batchsize, shuffle=False)

class GradientAttack():
        def __init__(self, loss, epsilon):
            self.loss = loss
            self.epsilon = epsilon

        def forward(self, x, y_true, model):
            x_adv = x
           
            y = model.forward(x_adv)
            J = self.loss(y,y_true)

            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)

            x_grad = torch.autograd.grad(J, x_adv)[0]
            x_adv = x + self.epsilon*x_grad.sign_()
            x_adv = torch.clamp(x_adv, 0, 1)

            return x_adv
                         
adv_attack = GradientAttack(loss_metric, 0.1)



def train_eval(verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for i, (images,labels) in enumerate(train_loader):
        print(images.shape)
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        #images = images.view(-1, 28*28)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        loss_sum += loss_metric(outputs,labels)
       
    if verbose:
        print('Train accuracy: %f %%' % (100 * correct / total))
        print('Train loss: %f' % (loss_sum.cpu().data.numpy().item() / total))

    return 100.0 * correct / total, loss_sum.cpu().data.numpy().item() / total
   
def test_eval(verbose = 1):
    correct = 0
    adv_correct = 0
    total = 0
    loss_sum = 0
    for i, (images,labels) in enumerate(test_loader):
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        #images = images.view(-1, 28*28)
        adv_outputs = net(images+(patch).cuda())
        _, adv_predicted = torch.max(adv_outputs.data, 1)
        total += labels.size(0)
        adv_correct += (adv_predicted.float() == labels.float()).sum()
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()
        #loss_sum += loss_metric(outputs,labels)

    if verbose:
        print('Test accuracy: %f %%' % (100 * correct / total))
        print('adv Test accuracy: %f %%' % (100 * adv_correct / total))
        #print('Test loss: %f' % (loss_sum.cpu().data.numpy().item() / total))

    return 100.0 * correct / total#, loss_sum.cpu().data.numpy().item() / total

def test_eval_adv(verbose = 1):
    correct = 0
    total = 0
    loss_sum = 0
    for i, (images,labels) in enumerate(train_loader):
        if gpu_boole:
            images, labels = images.cuda(), labels.cuda()
        #images = images.view(-1, 28*28)
        images = Variable(images, requires_grad=False)
        images = adv_attack.forward(images, Variable(labels), net)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.float() == labels.float()).sum()

        #loss_sum += loss_metric(outputs,labels)

    if verbose:
        print('Test accuracy adversarial: %f %%' % (100 * correct / total))
       # print('Test loss adversarial: %f' % (loss_sum.cpu().data.numpy().item() / total))

    return 100.0 * correct / total#, loss_sum.cpu().data.numpy().item() / total


#defining batch train loss recording arrays for later visualization/plotting:
loss_batch_store = []

print("Starting Training")

unet_pre_trained = torch.load('./detection.pth',map_location='cpu')
model_dict = net.state_dict()
for name, param in net.named_parameters():
         model_dict[name].copy_(unet_pre_trained[name])
         #param.requires_grad = True


#training loop:
for epoch in range(epochs):
  time1 = time.time() #timekeeping
  total_loss = 0
  total_det_loss = 0
  total_mse = 0
  for i, (x,y) in enumerate(train_loader):
          if gpu_boole:
            adv_x = (x+patch).cuda()
            y = y.cuda()
            x = x.cuda()
          else:
            adv_x = x+patch
            x = x
            y = y
          if i > 0 or epoch > 0:
            optimizer.zero_grad()
          outputs = net.forward(adv_x)
          mse_loss  = MSE_loss(adv_x,x)
          loss = mse_loss-loss_metric(outputs,y)
          det_loss = loss_metric(outputs,y)
          total_loss += loss.cpu().data.numpy().item()  
          total_mse += mse_loss.cpu().data.numpy().item()
          total_det_loss += det_loss.cpu().data.numpy().item()
          loss.backward()
#          if i > 0 or epoch > 0:
#            loss_batch_store.append(loss.cpu().data.numpy().item())

          ##performing update:x
#          optimizer.step()
          optimizer_adv.step()
  print("Epoch",epoch+1,':')
  print('total loss:',str(total_loss))
  print('total det loss:',str(total_det_loss))
  print('mse loss:',str(total_mse))
  test_perc = test_eval()
  #train_perc, train_loss = train_eval()
  #test_perc, test_loss = test_eval()
#  test_eval_adv()

#  time2 = time.time() #timekeeping
#  print('Elapsed time for epoch:',time2 - time1,'s')
#  print('ETA of completion:',(time2 - time1)*(epochs - epoch - 1)/60,'minutes')
#  print()
#
#  correct = 0
#  adv_correct = 0
#  total = 0
#  adv_total = 0
#  for i, (images,labels) in enumerate(test_loader):
#        if gpu_boole:
#            images, labels = images.cuda(), labels.cuda()
#        #images = images.view(-1, 28*28)
#        outputs = net(images)
#        _, predicted = torch.max(outputs.data, 1)
#        total += labels.size(0)
#        correct += (predicted.float() == labels.float()).sum()
#       
#        adv_outputs = net(images+patch)
#        _, adv_predicted = torch.max(adv_outputs.data, 1)
#        adv_total += labels.size(0)
#        adv_correct += (adv_predicted.float() == labels.float()).sum()
#    #loss_sum += loss_metric(outputs,labels)
#        print('Test accuracy: %f %%' % (100 * correct / total))
#        print('adv_test accuracy: %f %%' % (100 * adv_correct / total))
## Plotting batch-wise train loss curve:
#plt.plot(loss_batch_store, '-o', label = 'train_loss', color = 'blue')
#plt.xlabel('Minibatch Number')
#plt.ylabel('Sample-wise Loss At Last minibatch')
#plt.legend()
#plt.show()
#torch.save(net.state_dict(), '/home-net/home-4/span21@jhu.edu/Shaoyan_Pan/quicktest_cnn/detection.pth')