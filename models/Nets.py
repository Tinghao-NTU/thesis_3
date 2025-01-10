#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()


class CNNEMnist(nn.Module):
    def __init__(self):
        super(CNNEMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 12, 5, 1)
        self.fc1 = nn.Linear(192, 47)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 192)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_out)


    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        return F.log_softmax(x, dim=1)


class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, 1)
        self.conv2 = nn.Conv2d(5, 10, 5, 1)
        self.fc1 = nn.Linear(160, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 160)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, 5, 1)
        self.conv2 = nn.Conv2d(15, 28, 5, 1)
        self.fc1 = nn.Linear(700, 300)
        self.fc2 = nn.Linear(300, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 700)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    
class CNNMnist_Compare(nn.Module):
    def __init__(self):
        super(CNNMnist_Compare, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1)
        self.conv2 = nn.Conv2d(15, 28, 5, 1)
        self.fc1 = nn.Linear(448, 224)
        self.fc2 = nn.Linear(224, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 448)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class CNNMnist_Compare3(nn.Module):
    def __init__(self):
        super(CNNMnist_Compare3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 2, 1)
        self.conv2 = nn.Conv2d(20, 40, 2, 1)
        self.conv3 = nn.Conv2d(40, 80, 2, 1)
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return F.log_softmax(x, dim=1)
    
class CNNMnist_Compare4(nn.Module):
    def __init__(self):
        super(CNNMnist_Compare4, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 2, 1)
        self.conv2 = nn.Conv2d(20, 40, 2, 1)
        self.conv3 = nn.Conv2d(40, 80, 2, 1)
        self.conv4 = nn.Conv2d(80, 160, 2, 1)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
#         x = F.max_pool2d(x, 2, 2)        
        print(np.shape(x))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)        
        print(np.shape(x))
        return F.log_softmax(x, dim=1)   

class CNNMnist_Compare5(nn.Module):
    def __init__(self):
        super(CNNMnist_Compare, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5, 1)
        self.conv2 = nn.Conv2d(15, 28, 5, 1)
        self.fc1 = nn.Linear(448, 224)
        self.fc2 = nn.Linear(224, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 448)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class CNNCifar_Compare(nn.Module):
    def __init__(self):
        super(CNNCifar_Compare, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 14, 5)
        self.fc1 = nn.Linear(350, 150)
        self.fc2 = nn.Linear(150, 10)
        _initNetParams()
    def _initNetParams(self):
        '''Init net parameters.'''
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=1e-3)
                if m.bias:
                    init.constant(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 350)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class CNNCifar3(nn.Module):
    def __init__(self, args):
        super(CNNCifar3, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 2, 1)
        self.conv2 = nn.Conv2d(10, 15, 2, 1)
        self.conv3 = nn.Conv2d(15, 28, 2, 1)
#         self.conv4 = nn.Conv2d(15, 28, 5, 1)
        self.fc1 = nn.Linear(252, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)
#         self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)        
        x = x.view(-1,252)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
class CNNCifar4(nn.Module):
    def __init__(self, args):
        super(CNNCifar4, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 2, 1)
        self.conv2 = nn.Conv2d(10, 15, 2, 1)
        self.conv3 = nn.Conv2d(15, 20, 2, 1)
        self.conv4 = nn.Conv2d(20, 30, 2, 1)
        self.fc1 = nn.Linear(30, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)     
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)                   
        x = x.view(-1,30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)    
class CNNFashion(nn.Module):
    def __init__(self):
        super(CNNFashion, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 12, 5, 1)
        self.fc1 = nn.Linear(192, 80)
        self.fc2 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

