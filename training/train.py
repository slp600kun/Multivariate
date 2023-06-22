#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 07:44:50 2021

@author: shakeel
"""

import sys
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os 
from glob import glob
import pandas as pd 
import scipy.io as sc
import numpy as np
import datetime as dt
#from tqdm import tqdm
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

"""
class GEOEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.CNN_block_1 = ConvLayer2D(in_channels=1, out_channels=8, kernel=5, output_height=499, output_width=48)
        self.CNN_block_2 = ConvLayer2D(in_channels=8, out_channels=16, kernel=5, output_height=247, output_width=22)
        self.CNN_block_3 = ConvLayer2D(in_channels=16, out_channels=32, kernel=3, output_height=122, output_width=10)
        self.CNN_block_4 = ConvLayer2D(in_channels=32, out_channels=32, kernel=3, output_height=60, output_width=4)
        self.CNN_block_5 = ConvLayer2D(in_channels=32, out_channels=32, kernel=3, output_height=60, output_width=4)
        self.dense_1 = nn.Linear(7680, 50)
        self.dense_2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.CNN_block_1(input)
        output = self.CNN_block_2(output)
        output = self.CNN_block_3(output)
        output = self.CNN_block_4(output)
        #output = self.CNN_block_5(output)
        output = output.view(output.size(0), -1)
        output = self.relu(self.dense_1(output))
        output = self.sigmoid(self.dense_2(output))
        return output
'''
class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, output_height, output_width):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('max_pool', nn.AdaptiveMaxPool2d(output_size=(output_height,output_width)))
        self.add_module('drop', nn.Dropout2d(0.3))

    def forward(self, x):
        return super().forward(x)
"""
class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, output_height, output_width):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('max_pool', nn.AdaptiveMaxPool2d(output_size=(output_height,output_width)))
        self.add_module('drop', nn.Dropout2d(0.3))

    def forward(self, x):
        return super().forward(x)

class windEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.CNN_block_1 = ConvLayer2D(in_channels=1, out_channels=8, kernel=1, output_height=1, output_width=15)
        self.CNN_block_2 = ConvLayer2D(in_channels=8, out_channels=16, kernel=1, output_height=1, output_width=8)
        self.CNN_block_3 = ConvLayer2D(in_channels=16, out_channels=32, kernel=1, output_height=1, output_width=4)
        self.CNN_block_4 = ConvLayer2D(in_channels=32, out_channels=32, kernel=1, output_height=1, output_width=2)
        
    def forward(self, input):
        output = self.CNN_block_1(input)
        output = self.CNN_block_2(output)
        output = self.CNN_block_3(output)
        output = self.CNN_block_4(output)
        output = output.view(output.size(0), -1)
        return output

class CombinedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.gauss_enc = windEncoder()
        self.wind_enc = windEncoder()
        #self.dense_1 = nn.Linear(15360, 50)
        #self.dense_2 = nn.Linear(50, 1)
        self.dense_1 = nn.Linear(500, 250)
        self.dense_2 = nn.Linear(250, 1)  
        self.dense_3 = nn.Linear(500, 250)
        self.dense_4 = nn.Linear(250, 2)        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #self.softmax = F.log_softmax(net_output[0], -1)

        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.wind_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        #距離学習に対する出力
        output_one_dim = self.relu(self.dense_1(output))
        output_one_dim = self.sigmoid(self.dense_2(output_one_dim))

        #識別学習に対する出力
        output_two_dim = self.relu(self.dense_3(output))
        output_two_dim = self.sigmoid(self.dense_4(output_two_dim))        
        return output_one_dim, output_two_dim


class WindEncoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.LSTM_block_1 = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,batch_first=True,dropout=0.2)
        self.LSTM_block_2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=2,batch_first=True,dropout=0.2)
        self.LSTM_block_3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=2,batch_first=True,dropout=0.2)
        self.bn_block_1 = nn.BatchNorm1d(128)
        self.bn_block_2 = nn.BatchNorm1d(64)
        self.bn_block_3 = nn.BatchNorm1d(32)

    def forward(self, input):
        output,_ = self.LSTM_block_1(input)
        output = output.permute(0, 2, 1)
        output = self.bn_block_1(output)
        output = output.permute(0, 2, 1)

        output,_ = self.LSTM_block_2(output)
        output = output.permute(0, 2, 1)
        output = self.bn_block_2(output)
        output = output.permute(0, 2, 1)

        #output,_ = self.LSTM_block_3(output)
        #output = output.permute(0, 2, 1)
        #output = self.bn_block_3(output)
        #output = output.permute(0, 2, 1)

        output = self.LSTM_block_3(output)[0][:,-1,:]
        output = self.bn_block_3(output)
        
        return output
    
class CombinedEncoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.gauss_enc = WindEncoderLSTM()
        self.wind_enc = WindEncoderLSTM()
        self.dense_1 = nn.Linear(64, 32)
        self.dense_2 = nn.Linear(32, 1)
        self.dense_3 = nn.Linear(32, 2)
        self.bn_block_4 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.gauss_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        output = self.dense_1(output)
        output = self.relu(output)
        output = self.dropout_1(output)
        output = self.bn_block_4(output)

        #距離学習に対する出力
        output_one_dim = self.dense_2(output)
        output_one_dim = self.sigmoid(output_one_dim)
        #識別学習に対する出力
        output_two_dim = self.dense_3(output)
        output_two_dim = self.sigmoid(output_two_dim)

        return output_one_dim, output_two_dim
    
class DummyDataset(Dataset):
    """
    This class should contain complete dataset  in init 
    """

    def __init__(self, genuine_gauss,genuine_wind,forged_gauss,forged_wind,labels):
        super().__init__()
        self.true_gauss = genuine_gauss
        self.true_wind = genuine_wind
        self.wrong_gauss = forged_gauss
        self.wrong_wind = forged_wind
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        true_gauss_tensor = torch.tensor(self.true_gauss[idx],dtype=torch.float)
        true_wind_tensor = torch.tensor(self.true_wind[idx],dtype=torch.float) # this is complete dataset

        wrong_gauss_tensor = torch.tensor(self.wrong_gauss[idx],dtype=torch.float)
        wrong_wind_tensor = torch.tensor(self.wrong_wind[idx],dtype=torch.float)
        #convert_label = np.array(label[idx])
        labels = torch.tensor(self.labels[idx],dtype=torch.float)
        return true_gauss_tensor,true_wind_tensor,wrong_gauss_tensor,wrong_wind_tensor,labels
    
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1):
        super().__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + 1e-6) #euclidean distance
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / x0.size()[0]
        return loss, mdist
    
class SVM_for_two_dim(torch.nn.Module):
    """
    SVM machine for one dim data
    """
    def __init__(self):
        super(SVM_for_two_dim, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        return x
