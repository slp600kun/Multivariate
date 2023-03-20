#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 07:44:50 2021

@author: shakeel
"""

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
        self.CNN_block_1 = ConvLayer2D(in_channels=1, out_channels=8, kernel=5, output_height=499, output_width=24)
        self.CNN_block_2 = ConvLayer2D(in_channels=8, out_channels=16, kernel=3, output_height=247, output_width=11)
        self.CNN_block_3 = ConvLayer2D(in_channels=16, out_channels=32, kernel=3, output_height=122, output_width=4)
        self.CNN_block_4 = ConvLayer2D(in_channels=32, out_channels=32, kernel=3, output_height=60, output_width=4)
        
    def forward(self, input):
        output = self.CNN_block_1(input)
        output = self.CNN_block_2(output)
        output = self.CNN_block_3(output)
        output = self.CNN_block_4(output)
        output = output.view(output.size(0), -1)
        return output

'''
class windEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.CNN_block_1 = ConvLayer2D(in_channels=1, out_channels=8, kernel=5, output_height=499, output_width=24)
        self.CNN_block_2 = ConvLayer2D(in_channels=8, out_channels=16, kernel=3, output_height=247, output_width=11)
        self.CNN_block_3 = ConvLayer2D(in_channels=16, out_channels=32, kernel=3, output_height=122, output_width=4)
        self.CNN_block_4 = ConvLayer2D(in_channels=32, out_channels=32, kernel=3, output_height=60, output_width=4)
        #self.CNN_block_5 = ConvLayer2D(in_channels=32, out_channels=32, kernel=3, output_height=60, output_width=4)

    def forward(self, input):
        output = self.CNN_block_1(input)
        output = self.CNN_block_2(output)
        output = self.CNN_block_3(output)   
        output = self.CNN_block_4(output)
        #output = self.CNN_block_5(output)
        output = output.view(output.size(0), -1)
        return output
'''    
    
class CombinedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.gauss_enc = windEncoder()
        self.wind_enc = windEncoder()
        self.dense_1 = nn.Linear(15360, 50)
        self.dense_2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        #self.softmax = F.log_softmax(net_output[0], -1)

        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.wind_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        output = self.relu(self.dense_1(output))
        output = self.sigmoid(self.dense_2(output))
        return output

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
        true_gauss_tensor = torch.tensor(true_gauss[idx],dtype=torch.float)
        true_wind_tensor = torch.tensor(true_wind[idx],dtype=torch.float) # this is complete dataset

        wrong_gauss_tensor = torch.tensor(wrong_gauss[idx],dtype=torch.float)
        wrong_wind_tensor = torch.tensor(wrong_wind[idx],dtype=torch.float)
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
    
datadir = "train-npy/"
checkpoints_dir = "checkpoints/"
logs_dir = "logs/"

#true_gauss = np.load(datadir + 'gauss_a.npy')
#true_wind = np.load(datadir + 'wind_a.npy')
#wrong_gauss = np.load(datadir + 'gauss_b.npy')
#wrong_wind = np.load(datadir + 'wind_b.npy')
#label = np.load(datadir + 'labels.npy')

#true_gauss_train = true_gauss[0:31999]
#true_wind_train = true_wind[0:31999] 
#wrong_gauss_train = wrong_gauss[0:31999]
#wrong_wind_train = wrong_wind[0:31999]
#label_train = label[0:31999]

#true_gauss_val = true_gauss[32000:40000]
#true_wind_val = true_wind[32000:40000]
#wrong_gauss_val = wrong_gauss[32000:40000]
#wrong_wind_val = wrong_wind[32000:40000]
#label_val = label[32000:40000]wind_b.npy

true_gauss = np.load(datadir + 'gauss_a_set_4.npy')
true_wind = np.load(datadir + 'wind_a_set_4.npy')
wrong_gauss = np.load(datadir + 'gauss_b_set_4.npy')
wrong_wind = np.load(datadir + 'wind_b_set_4.npy')
label = np.load(datadir + 'labels_set_4.npy')

"""
true_gauss_train = true_gauss[0:127999]
true_wind_train = true_wind[0:127999] 
wrong_gauss_train = wrong_gauss[0:127999]
wrong_wind_train = wrong_wind[0:127999]
label_train = label[0:127999]

true_gauss_val = true_gauss[128000:160000]
true_wind_val = true_wind[128000:160000]
wrong_gauss_val = wrong_gauss[128000:160000]
wrong_wind_val = wrong_wind[128000:160000]
label_val = label[128000:160000]

true_gauss_train, true_gauss_val, label_train, label_val = train_test_split(true_gauss, label, test_size=0.2, shuffle=False)
true_wind_train, true_wind_val, label_train2, label_val2 = train_test_split(true_wind, label, test_size=0.2, shuffle=False)
wrong_gauss_train, wrong_gauss_val, label_train3, label_val3 = train_test_split(wrong_gauss, label, test_size=0.2, shuffle=False)
wrong_wind_train, wrong_wind_val, label_train4, label_val4 = train_test_split(wrong_wind, label, test_size=0.2, shuffle=False)
"""

traindataset = DummyDataset(true_gauss[0:6],true_wind[0:6],wrong_gauss[0:6],
                       wrong_wind[0:6],label[0:6])

valdataset = DummyDataset(true_gauss[7:9],true_wind[7:9],wrong_gauss[7:9],
                       wrong_wind[7:9],label[7:9])

epochs = 50
batch_size = 2
train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle=True)

#test_dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loss_fn = nn.CosineEmbeddingLoss().to(device)
loss_fn = ContrastiveLoss().to(device)
model = CombinedEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
model.train() 
torch.set_grad_enabled(True)
print("STARING TO TRAIN MODEL")
train_loss_list = []
val_acc_list = []
file1 = open(logs_dir + "training_accuracies.txt","w")
file2 = open(logs_dir + 'validation_accuracies.txt','w')
for epoch in range(1, epochs+1):
    model.train()
    steps_losses = []
    steps_accu = []
    model_checkpoints = checkpoints_dir + "model_" + str(epoch) + ".pt"
    for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        optimizer.zero_grad() 
        true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 1)
        true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 1)
        wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 1)
        wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 1)

        genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
        forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))
        loss,y_pred = loss_fn(genuine_output, forged_output, labels.to(device))
        steps_losses.append(loss.cpu().detach().numpy())
        train_loss_list.append(loss)
        prediction = (y_pred.cpu().detach().numpy()>0.4).astype(np.int)
        accuracy = accuracy_score(labels,prediction)
        steps_accu.append(accuracy)
        loss.backward()
        optimizer.step()

    now_time = dt.datetime.now()
    print(f"EPOCH {epoch}| Train:  loss {np.mean(steps_losses)}| accuracy {np.mean(steps_accu)} {now_time}")
    file1.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_losses)), "train_accuracy", str(np.mean(steps_accu)), now_time))
    torch.save(model.state_dict(),model_checkpoints)
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 1)
            true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 1)
            wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 1)
            wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 1)

            genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
            forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))
            loss,y_pred = loss_fn(genuine_output, forged_output, labels.to(device))
            prediction = (y_pred.cpu().detach().numpy()>0.4).astype(np.int)
            accuracy = accuracy_score(labels,prediction)
            val_acc_list.append(accuracy)
            steps_accu.append(accuracy)
            steps_losses.append(loss.cpu().numpy())
        print(f"EPOCH {epoch}| Validation:  loss {np.mean(steps_losses)}| accuracy {np.mean(steps_accu)} {now_time}")
        file2.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "val_loss", str(np.mean(steps_losses)), "val_accuracy", str(np.mean(steps_accu)), str(now_time)))
file1.close()
file2.close()

