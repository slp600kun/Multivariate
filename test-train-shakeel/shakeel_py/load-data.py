#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 02:49:02 2021

@author: shakeel
"""
import torch.utils.data as Data
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
#from tqdm import tqdm
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class DummyDataset(Dataset):

    def __init__(self, genuine_geo1,genuine_audio1,forged_geo1,forged_audio1,labels1):
        super().__init__()
        self.true_geos = genuine_geo1
        self.true_audios = genuine_audio1
        self.wrong_geos = forged_geo1
        self.wrong_audios = forged_audio1
        self.labels = labels1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        temp1 = sc.loadmat(data_di + self.true_geos.iloc[idx,0])
        feat1 = np.array(temp1['feat']).reshape(999,50)
        true_geo_tensor = torch.tensor(feat1,dtype=torch.float)
        
        temp2 = sc.loadmat(data_di + self.true_audios.iloc[idx,0])
        feat2 = np.array(temp2['feat']).reshape(999,50)
        true_audio_tensor = torch.tensor(feat2,dtype=torch.float) # this is complete dataset

        temp3 = sc.loadmat(data_di + self.wrong_geos.iloc[idx,0])
        feat3 = np.array(temp3['feat']).reshape(999,50)
        wrong_geo_tensor = torch.tensor(feat3,dtype=torch.float)
        
        temp4 = sc.loadmat(data_di + self.wrong_audios.iloc[idx,0])
        feat4 = np.array(temp4['feat']).reshape(999,50)
        wrong_audio_tensor = torch.tensor(feat4,dtype=torch.float)
        #convert_label = np.array(label[idx])
        labels = torch.tensor(self.labels.iloc[idx,0],dtype=torch.float)
        return true_audio_tensor,true_geo_tensor,wrong_audio_tensor,wrong_geo_tensor,labels
    
data_di = "/misc/export3/shakeel/multimodal/algo-all-data/train-data/new-features-999-50/data/"
genuine_geo = pd.read_csv(data_di + "geo_a.csv", header = None)
genuine_audio = pd.read_csv(data_di + "aud_a.csv", header = None)
forged_geo = pd.read_csv(data_di + "geo_b.csv", header = None)
forged_audio = pd.read_csv(data_di + "aud_b.csv", header = None)
labels = pd.read_csv(data_di + "label.csv", header = None)

true_geos_train, true_geos_val, label_train, label_val = train_test_split(genuine_geo, labels, test_size=0.2, shuffle=False)
true_audios_train, true_audios_val, label_train2, label_val2 = train_test_split(genuine_audio, labels, test_size=0.2, shuffle=False)
wrong_geos_train, wrong_geos_val, label_train3, label_val3 = train_test_split(forged_geo, labels, test_size=0.2, shuffle=False)
wrong_audios_train, wrong_audios_val, label_train4, label_val4 = train_test_split(forged_audio, labels, test_size=0.2, shuffle=False)

train_dataset = DummyDataset(true_geos_train,true_audios_train,wrong_geos_train,wrong_audios_train,label_train)
training_data = DataLoader(dataset=train_dataset, batch_size=100,num_workers=1)

valid_dataset = DummyDataset(true_geos_val,true_audios_val,wrong_geos_val,wrong_audios_val,label_val)
valid_data = DataLoader(dataset=valid_dataset, batch_size=100,num_workers=1)

"""
landmarks_frame = pd.read_csv(genuine_audio, header = None)
landmarks = landmarks_frame.iloc[1620,0]
temp1 = sc.loadmat(data_di + landmarks)
feat1 = np.array(temp1['feat']).reshape(999,50)
"""