# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 00:08:43 2019

@author: pan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 09:42:02 2019

@author: span21
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load('HW4_Prob1_data.npy')
train_data = data[:50000,:]
test_data = data[50000:,:]
num_parameter = 2
learning_rate = 1
deeper_learning_rate = 1.5
epochs = 400
filter_number = 10
def exp_deri(x):
    return np.exp(x)/((1+np.exp(x))**2)

def vector_transpose(x):
    width,height = x.shape
    return np.reshape(x,[height,width])

def sigmoid(x):
    y = np.exp(x)
    return y/(1+y)

class train_deeper_MLP():
    def __init__(self, num_parameter, learning_rate, epochs, filter_number, num_neruon = 10):
        self.learning_rate = learning_rate
        self.first_layer_first_weights = np.random.normal(0,1,[filter_number,num_parameter])
        self.first_layer_first_bias = np.random.normal(0,1,[filter_number,1])
        self.first_layer_second_weights = np.random.normal(0,1,[filter_number,num_parameter])
        self.first_layer_second_bias = np.random.normal(0,1,[filter_number,1])
       
        self.second_layer_first_weights = np.random.normal(0,1,[2,filter_number])
        self.second_layer_second_weights = np.random.normal(0,1,[2,filter_number])
        self.second_layer_first_bias = np.random.normal(0,1,[2,1])
        self.second_layer_second_bias = np.random.normal(0,1,[2,1])
       
        self.third_layer_first_weights = np.random.normal(0,1,[1,2])
        self.third_layer_second_weights = np.random.normal(0,1,[1,2])        
        self.third_layer_first_bias = np.random.normal(0,1,[1,1])        
        self.third_layer_second_bias = np.random.normal(0,1,[1,1])        
       
        self.final_layer_weight = np.random.normal(0,1,[1,2])
        self.final_layer_bias = np.random.normal(0,1,[1,1])
        self.num_parameter = num_parameter
        self.epochs = epochs
        self.num_neruon = num_neruon

   
    def training(self, train_data, train_label,test_dataset,testing_label):
        loss_list = []
        acc_list = []
        for ind in range(self.epochs):
            total_d_final_layer_weight = np.zeros([1,2])
            total_d_final_layer_bias = np.zeros([1,1])
           
            total_d_third_layer_first_weight = np.zeros([1,2])
            total_d_third_layer_first_bias = np.zeros([1,1])
            total_d_third_layer_second_weight = np.zeros([1,2])
            total_d_third_layer_second_bias = np.zeros([1,1])
           
            total_d_second_layer_first_weight = np.zeros([2,filter_number])
            total_d_second_layer_first_bias = np.zeros([2,1])
            total_d_second_layer_second_weight = np.zeros([2,filter_number])
            total_d_second_layer_second_bias = np.zeros([2,1])
           
            total_d_first_layer_first_weight = np.zeros([filter_number,2])
            total_d_first_layer_first_bias = np.zeros([filter_number,1])
            total_d_first_layer_second_weight = np.zeros([filter_number,2])
            total_d_first_layer_second_bias = np.zeros([filter_number,1])
           
            loss = 0
            for jnd in range(train_data.shape[0]):
                data = np.reshape(train_data[jnd,:],[2,1])
                first_layer_output1 = np.matmul(self.first_layer_first_weights,data) + self.first_layer_first_bias
                first_layer_output1_acitivated = sigmoid(first_layer_output1)
                first_layer_output2 = np.matmul(self.first_layer_second_weights,data) + self.first_layer_second_bias
                first_layer_output2_acitivated = sigmoid(first_layer_output2)
               
                second_layer_output1 = np.matmul(self.second_layer_first_weights,first_layer_output1_acitivated) + self.second_layer_first_bias
                second_layer_output1_acitivated = sigmoid(second_layer_output1)
                second_layer_output2 = np.matmul(self.second_layer_second_weights,first_layer_output2_acitivated) + self.second_layer_second_bias
                second_layer_output2_acitivated = sigmoid(second_layer_output2)
               
                third_layer_output1 = np.matmul(self.third_layer_first_weights,second_layer_output1_acitivated) + self.third_layer_first_bias
                third_layer_output1_acitivated = sigmoid(third_layer_output1)
                third_layer_output2 = np.matmul(self.third_layer_second_weights,second_layer_output2_acitivated) + self.third_layer_second_bias
                third_layer_output2_acitivated = sigmoid(third_layer_output2)
               
                third_layer_output = np.concatenate([third_layer_output1_acitivated,third_layer_output2_acitivated])
               
                final_layer_output = np.matmul(self.final_layer_weight,third_layer_output)+self.final_layer_bias
                final_layer_output_acitivated = sigmoid(final_layer_output)
                loss += -train_label[jnd]*np.log(final_layer_output_acitivated+1e-6)-(1-train_label[jnd])*np.log(1-final_layer_output_acitivated+1e-6)
               
                d_final_layer_output = -train_label[jnd]/final_layer_output_acitivated + (1-train_label[jnd])/(1-final_layer_output_acitivated)
               
                d_final_layer_weight = d_final_layer_output*exp_deri(final_layer_output)*third_layer_output.T
                d_final_layer_bias = d_final_layer_output*exp_deri(final_layer_output)                
                d_third_layer_output = (self.final_layer_weight).T*(d_final_layer_output*exp_deri(final_layer_output))
               
                total_d_final_layer_weight += d_final_layer_weight
                total_d_final_layer_bias += d_final_layer_bias
               
                d_third_layer_first_weight = d_third_layer_output[0]*exp_deri(third_layer_output1)*second_layer_output1_acitivated.T
                d_third_layer_first_bias = d_third_layer_output[0]*exp_deri(third_layer_output1)
                d_second_layer_output1 =  self.third_layer_first_weights.T*(d_third_layer_output[0]*exp_deri(third_layer_output1))
               
                d_third_layer_second_weight = d_third_layer_output[1]*exp_deri(third_layer_output2)*second_layer_output2_acitivated.T
                d_third_layer_second_bias = d_third_layer_output[1]*exp_deri(third_layer_output2)                
                d_second_layer_output2 = self.third_layer_second_weights.T*(d_third_layer_output[1]*exp_deri(third_layer_output2))
               
                total_d_third_layer_first_weight += d_third_layer_first_weight
                total_d_third_layer_first_bias += d_third_layer_first_bias
                total_d_third_layer_second_weight += d_third_layer_second_weight
                total_d_third_layer_second_bias += d_third_layer_second_bias
               
                d_second_layer_first_weight = d_second_layer_output1*exp_deri(second_layer_output1)*first_layer_output1_acitivated.T
                d_second_layer_first_bias = d_second_layer_output1*exp_deri(second_layer_output1)
                d_first_layer_output1 =  np.matmul((self.second_layer_first_weights).T,(d_second_layer_output1*exp_deri(second_layer_output1)))
               
                d_second_layer_second_weight = d_second_layer_output2*exp_deri(second_layer_output2)*first_layer_output2_acitivated.T
                d_second_layer_second_bias = d_second_layer_output2*exp_deri(second_layer_output2)                
                d_first_layer_output2 = np.matmul((self.second_layer_second_weights).T,(d_second_layer_output2*exp_deri(second_layer_output2)))
               
                total_d_second_layer_first_weight += d_second_layer_first_weight
                total_d_second_layer_first_bias += d_second_layer_first_bias
                total_d_second_layer_second_weight += d_second_layer_second_weight
                total_d_second_layer_second_bias += d_second_layer_second_bias
               
                d_first_layer_first_weight = d_first_layer_output1*exp_deri(first_layer_output1)*train_data[jnd,:].T
                d_first_layer_first_bias = d_first_layer_output1*exp_deri(first_layer_output1)
                d_first_layer_second_weight = d_first_layer_output2*exp_deri(first_layer_output2)*train_data[jnd,:].T
                d_first_layer_second_bias = d_first_layer_output2*exp_deri(first_layer_output2)
               
                total_d_first_layer_first_weight += d_first_layer_first_weight
                total_d_first_layer_first_bias += d_first_layer_first_bias
                total_d_first_layer_second_weight += d_first_layer_second_weight
                total_d_first_layer_second_bias += d_first_layer_second_bias
           
            loss_list.append(loss[0][0])
            acc,_ = self.accuracy(test_dataset,testing_label)
            acc_list.append(acc)
            print(loss[0][0])
            print(acc)
            print(ind)
            self.first_layer_first_weights = self.first_layer_first_weights - self.learning_rate*total_d_first_layer_first_weight/train_data.shape[0]
            self.first_layer_second_weights = self.first_layer_second_weights - self.learning_rate*total_d_first_layer_second_weight/train_data.shape[0]
            self.first_layer_first_bias = self.first_layer_first_bias - self.learning_rate*total_d_first_layer_first_bias/train_data.shape[0]
            self.first_layer_second_bias = self.first_layer_second_bias - self.learning_rate*total_d_first_layer_second_bias/train_data.shape[0]
            self.second_layer_first_weights = self.second_layer_first_weights - self.learning_rate*total_d_second_layer_first_weight/train_data.shape[0]
            self.second_layer_second_weights = self.second_layer_second_weights - self.learning_rate*total_d_second_layer_second_weight/train_data.shape[0]
            self.second_layer_first_bias = self.second_layer_first_bias - self.learning_rate*total_d_second_layer_first_bias/train_data.shape[0]
            self.second_layer_second_bias = self.second_layer_second_bias - self.learning_rate*total_d_second_layer_second_bias/train_data.shape[0]
            self.third_layer_first_weights = self.third_layer_first_weights - self.learning_rate*total_d_third_layer_first_weight/train_data.shape[0]
            self.third_layer_second_weights = self.third_layer_second_weights - self.learning_rate*total_d_third_layer_second_weight/train_data.shape[0]
            self.third_layer_first_bias = self.third_layer_first_bias - self.learning_rate*total_d_third_layer_first_bias/train_data.shape[0]
            self.third_layer_second_bias = self.third_layer_second_bias - self.learning_rate*total_d_third_layer_second_bias/train_data.shape[0]
            self.final_layer_weight = self.final_layer_weight-self.learning_rate*total_d_final_layer_weight/train_data.shape[0]
            self.final_layer_bias = self.final_layer_bias-self.learning_rate*total_d_final_layer_bias/train_data.shape[0]
        return loss_list,acc_list          

    def accuracy(self, data, label):
        count = 0
        final_decision = np.zeros(label.shape[0])
        for ind in range(label.shape[0]):
            final_layer_output = self.prediction_logit(data[ind,:])
            if final_layer_output>0.5:
                final_decision[ind] = 1
            else:
                final_decision[ind] = 0
            if final_decision[ind] == label[ind]:
                count += 1
        accuracy = count/label.shape[0]*100
        return accuracy, final_decision
   
    def prediction_logit(self, data):
        data = np.reshape(data,[2,1])
        first_layer_output1 = np.matmul(self.first_layer_first_weights,data) + self.first_layer_first_bias
        first_layer_output1_acitivated = sigmoid(first_layer_output1)
        first_layer_output2 = np.matmul(self.first_layer_second_weights,data) + self.first_layer_second_bias
        first_layer_output2_acitivated = sigmoid(first_layer_output2)
       
        second_layer_output1 = np.matmul(self.second_layer_first_weights,first_layer_output1_acitivated) + self.second_layer_first_bias
        second_layer_output1_acitivated = sigmoid(second_layer_output1)
        second_layer_output2 = np.matmul(self.second_layer_second_weights,first_layer_output2_acitivated) + self.second_layer_second_bias
        second_layer_output2_acitivated = sigmoid(second_layer_output2)
       
        third_layer_output1 = np.matmul(self.third_layer_first_weights,second_layer_output1_acitivated) + self.third_layer_first_bias
        third_layer_output1_acitivated = sigmoid(third_layer_output1)
        third_layer_output2 = np.matmul(self.third_layer_second_weights,second_layer_output2_acitivated) + self.third_layer_second_bias
        third_layer_output2_acitivated = sigmoid(third_layer_output2)
       
        third_layer_output = np.concatenate([third_layer_output1_acitivated,third_layer_output2_acitivated])
       
        final_layer_output = np.matmul(self.final_layer_weight,third_layer_output)+self.final_layer_bias
        final_layer_output_acitivated = sigmoid(final_layer_output)
        return final_layer_output_acitivated
   
class train_MLP():
    def __init__(self, num_parameter, learning_rate, epochs, filter_number, num_neruon = 10):
        self.learning_rate = learning_rate
        self.first_layer_first_weights = np.random.normal(0,1,[5,num_parameter])
        self.first_layer_first_bias = np.random.normal(0,1,[5,1])
        self.first_layer_second_weights = np.random.normal(0,1,[5,num_parameter])
        self.first_layer_second_bias = np.random.normal(0,1,[5,1])
       
        self.second_layer_first_weights = np.random.normal(0,1,[1,5])
        self.second_layer_second_weights = np.random.normal(0,1,[1,5])
        self.second_layer_first_bias = np.random.normal(0,1,[1,1])
        self.second_layer_second_bias = np.random.normal(0,1,[1,1])
       
        self.final_layer_weight = np.random.normal(0,1,[1,2])
        self.final_layer_bias = np.random.normal(0,1,[1,1])
        self.num_parameter = num_parameter
        self.epochs = epochs
        self.num_neruon = num_neruon

   
    def training(self, train_data, train_label,test_dataset,testing_label):
        loss_list = []
        acc_list = []
        for ind in range(self.epochs):
            total_d_final_layer_weight = np.zeros([1,2])
            total_d_final_layer_bias = np.zeros([1,1])
            total_d_second_layer_first_weight = np.zeros([1,5])
            total_d_second_layer_first_bias = np.zeros([1,1])
            total_d_second_layer_second_weight = np.zeros([1,5])
            total_d_second_layer_second_bias = np.zeros([1,1])
            total_d_first_layer_first_weight = np.zeros([5,self.num_parameter])
            total_d_first_layer_first_bias = np.zeros([5,1])
            total_d_first_layer_second_weight = np.zeros([5,self.num_parameter])
            total_d_first_layer_second_bias = np.zeros([5,1])
           
            loss = 0
            for jnd in range(train_data.shape[0]):
                data = np.reshape(train_data[jnd,:],[2,1])
                first_layer_output1 = np.matmul(self.first_layer_first_weights,data) + self.first_layer_first_bias
                first_layer_output1_acitivated = sigmoid(first_layer_output1)
                second_layer_output1 = np.matmul(self.second_layer_first_weights,first_layer_output1_acitivated) + self.second_layer_first_bias
                second_layer_output1_acitivated = sigmoid(second_layer_output1)
               
                first_layer_output2 = np.matmul(self.first_layer_second_weights,data) + self.first_layer_second_bias
                first_layer_output2_acitivated = sigmoid(first_layer_output2)
                second_layer_output2 = np.matmul(self.second_layer_second_weights,first_layer_output2_acitivated) + self.second_layer_second_bias
                second_layer_output2_acitivated = sigmoid(second_layer_output2)
               
                second_layer_output = np.concatenate([second_layer_output1_acitivated,second_layer_output2_acitivated])
               
                final_layer_output = np.matmul(self.final_layer_weight,second_layer_output)+self.final_layer_bias
                final_layer_output_acitivated = sigmoid(final_layer_output)
                loss += -train_label[jnd]*np.log(final_layer_output_acitivated+1e-6)-(1-train_label[jnd])*np.log(1-final_layer_output_acitivated+1e-6)
               
                d_final_layer_output = -train_label[jnd]/final_layer_output_acitivated + (1-train_label[jnd])/(1-final_layer_output_acitivated)
               
                d_final_layer_weight = d_final_layer_output*exp_deri(final_layer_output)*second_layer_output.T
                d_final_layer_bias = d_final_layer_output*exp_deri(final_layer_output)                
                d_second_layer_output = d_final_layer_output*self.final_layer_weight.T*exp_deri(final_layer_output)
               
                total_d_final_layer_weight += d_final_layer_weight
                total_d_final_layer_bias += d_final_layer_bias

                d_second_layer_first_weight = d_second_layer_output[0]*exp_deri(second_layer_output1)*first_layer_output1_acitivated.T
                d_second_layer_first_bias = d_second_layer_output[0]*exp_deri(second_layer_output1)
                d_first_layer_output1 =  self.second_layer_first_weights.T*(d_second_layer_output[0]*exp_deri(second_layer_output1))
               
                d_second_layer_second_weight = d_second_layer_output[1]*exp_deri(second_layer_output2)*first_layer_output2_acitivated.T
                d_second_layer_second_bias = d_second_layer_output[1]*exp_deri(second_layer_output2)                
                d_first_layer_output2 = self.second_layer_second_weights.T*(d_second_layer_output[1]*exp_deri(second_layer_output2))
               
                total_d_second_layer_first_weight += d_second_layer_first_weight
                total_d_second_layer_first_bias += d_second_layer_first_bias
                total_d_second_layer_second_weight += d_second_layer_second_weight
                total_d_second_layer_second_bias += d_second_layer_second_bias
               
                d_first_layer_first_weight = d_first_layer_output1*exp_deri(first_layer_output1)*train_data[jnd,:].T
                d_first_layer_first_bias = d_first_layer_output1*exp_deri(first_layer_output1)
                d_first_layer_second_weight = d_first_layer_output2*exp_deri(first_layer_output2)*train_data[jnd,:].T
                d_first_layer_second_bias = d_first_layer_output2*exp_deri(first_layer_output2)
               
                total_d_first_layer_first_weight += d_first_layer_first_weight
                total_d_first_layer_first_bias += d_first_layer_first_bias
                total_d_first_layer_second_weight += d_first_layer_second_weight
                total_d_first_layer_second_bias += d_first_layer_second_bias
           
            loss_list.append(loss[0][0])
            acc,_ = self.accuracy(test_dataset,testing_label)
            acc_list.append(acc)
            print(loss[0][0])
            print(acc)
            print(ind)
            self.first_layer_first_weights = self.first_layer_first_weights - self.learning_rate*total_d_first_layer_first_weight/train_data.shape[0]
            self.first_layer_second_weights = self.first_layer_second_weights - self.learning_rate*total_d_first_layer_second_weight/train_data.shape[0]
            self.first_layer_first_bias = self.first_layer_first_bias - self.learning_rate*total_d_first_layer_first_bias/train_data.shape[0]
            self.first_layer_second_bias = self.first_layer_second_bias - self.learning_rate*total_d_first_layer_second_bias/train_data.shape[0]
            self.second_layer_first_weights = self.second_layer_first_weights - self.learning_rate*total_d_second_layer_first_weight/train_data.shape[0]
            self.second_layer_second_weights = self.second_layer_second_weights - self.learning_rate*total_d_second_layer_second_weight/train_data.shape[0]
            self.second_layer_first_bias = self.second_layer_first_bias - self.learning_rate*total_d_second_layer_first_bias/train_data.shape[0]
            self.second_layer_second_bias = self.second_layer_second_bias - self.learning_rate*total_d_second_layer_second_bias/train_data.shape[0]
            self.final_layer_weight = self.final_layer_weight-self.learning_rate*total_d_final_layer_weight/train_data.shape[0]
            self.final_layer_bias = self.final_layer_bias-self.learning_rate*total_d_final_layer_bias/train_data.shape[0]
        return loss_list,acc_list

    def accuracy(self, data, label):
        count = 0
        final_decision = np.zeros(label.shape[0])
        for ind in range(label.shape[0]):
            final_layer_output = self.prediction_logit(data[ind,:])
            if final_layer_output>0.5:
                final_decision[ind] = 1
            else:
                final_decision[ind] = 0
            if final_decision[ind] == label[ind]:
                count += 1
        accuracy = count/label.shape[0]*100
        return accuracy, final_decision
   
    def prediction_logit(self, data):
        data = np.reshape(data,[2,1])
        first_layer_output1 = np.matmul(self.first_layer_first_weights,data) + self.first_layer_first_bias
        first_layer_output1_acitivated = 1/(1+np.exp(-first_layer_output1))
        first_layer_output2 = np.matmul(self.first_layer_second_weights,data) + self.first_layer_second_bias
        first_layer_output2_acitivated = 1/(1+np.exp(-first_layer_output2))
        second_layer_output1 = np.matmul(self.second_layer_first_weights,first_layer_output1_acitivated) + self.second_layer_first_bias
        second_layer_output1_acitivated = 1/(1+np.exp(-second_layer_output1))
        second_layer_output2 = np.matmul(self.second_layer_second_weights,(first_layer_output2_acitivated)) + self.second_layer_second_bias
        second_layer_output2_acitivated = 1/(1+np.exp(-second_layer_output2))
        second_layer_output = np.concatenate([second_layer_output1_acitivated,second_layer_output2_acitivated])
        final_layer_output = np.matmul(self.final_layer_weight,second_layer_output)+self.final_layer_bias
        final_layer_output_acitivated = 1/(1+np.exp(-final_layer_output))
        return final_layer_output_acitivated
   
class manual_MLP():
    def __init__(self, num_parameter, num_neruon = 10):
        self.weights = np.zeros([num_neruon,num_parameter])
        self.bias = np.zeros(num_neruon).T
        self.num_neruon = num_neruon
   
    def setup_W_and_B(self):
        #The first pentagon
        self.weights[0,:] = [1,-1]
        self.bias[0] = 500
        self.weights[1,:] = [2, 1]
        self.bias[1] = -1400
        self.weights[2,:] = [0, 1]
        self.bias[2] = -600
        self.weights[3,:] = [-2,1]
        self.bias[3] = 600
        self.weights[4,:] = [-1,-1]
        self.bias[4] = 1500
        #The second pentagon
        self.weights[5,:] = [1,-2]
        self.bias[5] = 700
        self.weights[6,:] = [1, 1]
        self.bias[6] = -500
        self.weights[7,:] = [0, 1]
        self.bias[7] = -200
        self.weights[8,:] = [-1,1]
        self.bias[8] = 500
        self.weights[9,:] = [-1,-2]
        self.bias[9] = 1700
               
    def prediction(self, data, label):
        self.setup_W_and_B()
        final_decision = np.zeros(label.shape[0])
        count = 0
        y = np.zeros(self.num_neruon)
        for ind in range(label.shape[0]):
            for knd in range(0,5):
                weight = self.weights[knd,:]
                output = np.dot(weight,(data[ind,:])) + self.bias[knd]
                y[knd] = 1 if output>0 else 0
                first_penta_decision = y[0:5].all()
            for knd in range(5,10):
                weight = self.weights[knd,:]
                output = np.dot(weight,(data[ind,:])) + self.bias[knd]
                y[knd] = 1 if output>0 else 0
            second_penta_decision = y[5:10].all()
            final_decision[ind] = int((first_penta_decision or second_penta_decision))
            if final_decision[ind] == label[ind]:
                count += 1
            else:
                print(str(data[ind,:]))
                print(str(label[ind]))
        accuracy = count/label.shape[0]*100
        return accuracy,final_decision    
   
#train_dataset = train_data[:,:2]
#label_dataset = train_data[:,2]
#test_dataset = test_data[:,:2]
#test_label_dataset = test_data[:,2]
#Prediction_manual_MLP = manual_MLP(num_parameter)
#Accuracy_manual_MLP, final_decisions = Prediction_manual_MLP.prediction(test_dataset,test_label_dataset)
#plt.figure(1)
#for i in range(test_data.shape[0]):
#    if final_decisions[i]==0:
#        plt.plot(test_data[i,0],test_data[i,1],',',color='blue',markersize=15)
#    elif final_decisions[i]==1:
#        plt.plot(test_data[i,0],test_data[i,1],',',color='green',markersize=15)
#plt.savefig('Q1A visualization')

train_dataset = (train_data[:,:2]-np.mean(train_data[:,:2],0))/np.std(train_data[:,:2],0)
test_dataset = (test_data[:,:2]-np.mean(test_data[:,:2],0))/np.std(test_data[:,:2],0)
label_dataset = train_data[:,2]
testing_label = test_data[:,2]
#Prediction_MLP = train_MLP(num_parameter, learning_rate, epochs,5)
#loss_list,acc_list = Prediction_MLP.training(train_dataset,label_dataset,test_dataset,testing_label)
#Accuracy_MLP, final_decisions = Prediction_MLP.accuracy(test_dataset,testing_label)
#plt.figure(2)
#for i in range(testing_label.shape[0]):
#    if final_decisions[i]==0:
#        plt.plot(test_data[i,0],test_data[i,1],',',color='b',markersize=15)
#    elif final_decisions[i]==1:
#        plt.plot(test_data[i,0],test_data[i,1],',',color='g',markersize=15)
#plt.savefig('Q1B visualization')
#
#plt.figure(3)
#plt.plot(acc_list)
#plt.ylabel('Accuracy in Q1b weak MLP')
#plt.xlabel('Epoch')
#plt.savefig('Q1b_acc.png')
#
#plt.figure(4)
#plt.plot(loss_list)
#plt.ylabel('Loss in Q1b weak MLP')
#plt.xlabel('Epoch')
#plt.savefig('Q1b_loss.png')


Prediction_larger_MLP = train_deeper_MLP(num_parameter, deeper_learning_rate, epochs,filter_number)
Advloss_list,Advacc_list = Prediction_larger_MLP.training(train_dataset,label_dataset,test_dataset,testing_label)
Accuracy_deeper_MLP, final_deeper_decisions = Prediction_larger_MLP.accuracy(test_dataset,testing_label)
plt.figure(5)
for i in range(testing_label.shape[0]):
    if final_deeper_decisions[i]==0:
        plt.plot(test_data[i,0],test_data[i,1],',',color='b',markersize=15)
    elif final_deeper_decisions[i]==1:
        plt.plot(test_data[i,0],test_data[i,1],',',color='g',markersize=15)
plt.savefig('Q1B visualization')

plt.figure(6)
plt.plot(Advacc_list)
plt.ylabel('Accuracy in Q1c strong MLP')
plt.xlabel('Epoch')
plt.savefig('Q1c_acc.png')

plt.figure(7)
plt.plot(Advloss_list)
plt.ylabel('Loss in Q1c strong MLP')
plt.xlabel('Epoch')
plt.savefig('Q1c_loss.png')


nihao = Advloss_list
henhao = Advacc_list