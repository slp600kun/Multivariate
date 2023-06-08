import sys
import os
import re
import numpy as np
from preprocess_data import preprocess_for_Siamese_Net
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader,TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tqdm.auto import tqdm
import datetime as dt
import random
import pandas as pd

class DummyDataset(Dataset):
    """
    This class should contain complete dataset  in init 
    """

    def __init__(self, genuine_gauss,genuine_wind,labels):
        super().__init__()
        self.true_gauss = genuine_gauss
        self.true_wind = genuine_wind
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        true_gauss_tensor = torch.tensor(self.true_gauss[idx],dtype=torch.float)
        true_wind_tensor = torch.tensor(self.true_wind[idx],dtype=torch.float) # this is complete dataset
        labels = self.labels[idx].to(torch.float)
        return true_gauss_tensor,true_wind_tensor,labels

class WindEncoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.LSTM_block_1 = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,batch_first=True,dropout=0.2)
        self.LSTM_block_2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,batch_first=True,dropout=0.2)
        self.LSTM_block_3 = nn.LSTM(input_size=256, hidden_size=64, num_layers=2,batch_first=True,dropout=0.2)
        self.bn_block_1 = nn.BatchNorm1d(128)
        self.bn_block_2 = nn.BatchNorm1d(256)
        self.bn_block_3 = nn.BatchNorm1d(64)

    def forward(self, input):
        input = torch.unsqueeze(input,dim=2)
        output,_ = self.LSTM_block_1(input)
        output = output.permute(0, 2, 1)
        output = self.bn_block_1(output)
        output = output.permute(0, 2, 1)

        output,_ = self.LSTM_block_2(output)
        output = output.permute(0, 2, 1)
        output = self.bn_block_2(output)
        output = output.permute(0, 2, 1)

        output = self.LSTM_block_3(output)[0][:,-1,:]
        output = self.bn_block_3(output)

        return output


class CombinedEncoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.gauss_enc = WindEncoderLSTM()
        self.wind_enc = WindEncoderLSTM()
        self.dense_1 = nn.Linear(128, 64)
        self.dense_2 = nn.Linear(64, 2)
        self.dense_3 = nn.Linear(64, 32)
        self.bn_block_4 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(False)
        self.dropout_1 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.gauss_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        
        output = self.dense_1(output)
        output = self.bn_block_4(output)
        output = self.relu(output)

        #距離学習に対する出力
        output_two_dim = self.dense_2(output)
    
        return output_two_dim

datadir = "data/train-npy/"
checkpoints_dir = "data/checkpoints/"
logs_dir = "data/logs/"

true_gauss = np.load(datadir + 'gauss_a_set.npy')
true_wind = np.load(datadir + 'wind_a_set.npy')
train_data_len = 3000000

scaler_gauss = StandardScaler()
scaler_wind = StandardScaler()

scaler_gauss.fit(true_gauss[0:train_data_len])
scaler_wind.fit(true_wind[0:train_data_len])

#テストデータ
test_true_gauss = np.load(datadir + 'test_gauss_a_set.npy')
test_true_wind = np.load(datadir + 'test_wind_a_set.npy')
test_label = np.load(datadir + 'test_labels.npy')

test_data_len = 50000
#識別学習に用いるone-hot表現のラベルを作成
one_hot_testlabels = torch.zeros(test_data_len, 2, dtype=torch.float)
for step, genuine_label in enumerate(test_label[:test_data_len][:,0]):
    if genuine_label == 1:
        one_hot_testlabels[step]=torch.tensor([1,0],dtype=torch.float)
    if genuine_label == 0:
        one_hot_testlabels[step]=torch.tensor([0,1],dtype=torch.float)

test_scaled_gauss = scaler_gauss.transform(test_true_gauss)
test_scaled_wind = scaler_wind.transform(test_true_wind)

testdataset = DummyDataset(test_scaled_gauss[0:test_data_len] ,test_scaled_wind[0:test_data_len] ,one_hot_testlabels[0:test_data_len])
batch_size = 1000
test_dataloader = DataLoader(testdataset, batch_size = batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CombinedEncoderLSTM()
model.to(device)
model_path = 'data/checkpoints/model_1.pt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()

test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []
with torch.no_grad():
    for steps, (true_gauss_tensor, true_wind_tensor, test_labels) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
        test_outputs = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))

        #calculate loss
        _, test_true_classes = test_labels.clone().max(dim=1)
        #calculate accuracy
        test_outputs_softmax = torch.softmax(test_outputs,dim=1)
        test_predicted_classes = torch.argmax(test_outputs_softmax, dim=1)
        # one-hot表現に変換
        test_accuracy = accuracy_score(test_predicted_classes.cpu(), test_true_classes.cpu())
        test_accuracies.append(test_accuracy)
        test_precision = precision_score(test_predicted_classes.cpu(), test_true_classes.cpu(), average='macro')
        test_precisions.append(test_precision)
        test_recall = recall_score(test_predicted_classes.cpu(), test_true_classes.cpu(), average='macro',zero_division=0)
        test_recalls.append(test_recall)        
        test_f1_score = f1_score(test_predicted_classes.cpu(), test_true_classes.cpu(), average='macro')
        test_f1_scores.append(test_f1_score)

acc = np.mean(test_accuracies)
prec = np.mean(test_precisions)
recall = np.mean(test_recalls)
f1 = np.mean(test_f1_scores)
print(f"steps Accuracy: {acc}")
print(f"steps precision: {prec}")
print(f"steps recall: {recall}")
print(f"steps f1 score: {f1}")
file3 = open(logs_dir + "test_LSTM1-classifier_metrics.txt","w")
file3.write("%s,%s,\n%s,%s,\n%s,%s,\n%s,%s,\n" %("Accuracy",str(acc),"Precision",str(prec),"Recall",str(recall),"F1 score",str(f1)))
file3.close()
    
