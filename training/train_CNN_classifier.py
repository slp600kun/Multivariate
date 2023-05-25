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
import random
#from tqdm import tqdm
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
        input = torch.unsqueeze(input,dim=1)
        input = torch.unsqueeze(input,dim=2)
        output = self.CNN_block_1(input)
        output = self.CNN_block_2(output)
        output = self.CNN_block_3(output)
        output = self.CNN_block_4(output)
        output = output.view(output.size(0), -1)
        return output
    
class CombinedEncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.gauss_enc = windEncoder()
        self.wind_enc = windEncoder()
        self.dense_1 = nn.Linear(128, 64)
        self.dense_2 = nn.Linear(64, 2)
        self.bn_block_4 = nn.BatchNorm1d(64)
        self.dropout_1 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.wind_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        output = self.relu(self.dense_1(output))
        output = self.dropout_1(output)
        output = self.bn_block_4(output)

        output = self.dense_2(output)
        return output

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
        true_gauss_tensor = torch.tensor(true_gauss[idx],dtype=torch.float)
        true_wind_tensor = torch.tensor(true_wind[idx],dtype=torch.float) # this is complete dataset

        labels = torch.tensor(self.labels.clone().detach()[idx],dtype=torch.float)
        return true_gauss_tensor,true_wind_tensor,labels
    
    
def generate_siamese_data(action_df:pd,
                          not_action_df:pd,
                          slide_length:int,
                          segment_data_length:int,
                          not_action_df_starting_point:int):
    
    """
    dfの対から必要なデータ列を選択し、一定の長さで
    """

    def slide_time_data(df:pd,slide_length:int,segment_data_length:int,
                            is_wind_vel_converted: bool = True,
                            is_temp_converted: bool = False,
                            is_humid_converted: bool = False,
                            is_gauss_converted: bool = True) ->list:
        """
        merged_dfから必要なデータ列を選択し、一定の長さの窓を指定した秒間隔でスライドさせた状態で出力する
        args: 
            - df (pd): 元のmerged_df
            - slide_length(int): スライド間隔(秒)
            - segment_data_length (int):データの分割窓長さ(秒)
            - is _wind_vel_converted (bool) :風速を出力に入れるか
            - is _temp_vel_converted (bool) :温度を出力に入れるか
            - is _humid_vel_converted (bool) :湿度を出力に入れるか
            - is _gauss_vel_converted (bool) :磁束密度を出力に入れるか

        return:
            - segmented_datafarme_array(list)選択したデータをスライド出力したもの
        """

        #データを選ぶ
        if is_wind_vel_converted == False:
            df = df.drop('V(m/s)', axis=1)
        
        if is_temp_converted == False:
            df = df.drop('T(C)', axis=1)
        
        if is_humid_converted == False:
            df = df.drop('H(%RH)', axis=1)
        
        if is_gauss_converted == False:
            df = df.drop('φ(mG)', axis=1)   
        
        #出力配列
        segmented_datafarme_array = []

        #segment
        for i in range(0, len(df), slide_length):
            segment_dataframe = df[i:i+segment_data_length]
            if segment_dataframe.shape == (segment_data_length,len(df.columns)):
                segmented_datafarme_array.append(segment_dataframe)

        return  segmented_datafarme_array
    
    #出力配列
    action_feat1 = []
    action_feat2 = []
    not_action_feat1 = []
    not_action_feat2 = []

    #分割されたdf配列(この状態だとデータフレームの配列になる)
    action_segment_data_list = slide_time_data(action_df,slide_length,segment_data_length)
    not_action_segment_data_list = slide_time_data(not_action_df[not_action_df_starting_point:not_action_df_starting_point+len(action_df)],
                                                                 slide_length,segment_data_length)

    #df先頭の特徴量のリスト
    feat_list = list(action_segment_data_list[0].columns.values.tolist())

    #各配列のdfを配列に変換
    for action_df,not_action_df in zip(action_segment_data_list,not_action_segment_data_list):
        action_feat1.append(action_df[feat_list[0]].values)
        not_action_feat1.append(not_action_df[feat_list[0]].values)
        action_feat2.append(action_df[feat_list[1]].values)
        not_action_feat2.append(not_action_df[feat_list[1]].values)
    return action_feat1,action_feat2,not_action_feat1,not_action_feat2 

def generate_npy_from_siamese_data(action_feat1:list,action_feat2:list,not_action_feat1:list,not_action_feat2:list):

    """
    siamese dataをラベリングし、npyファイルに出力する関数

    args:
        action_feat1(list):行動を取った時の特徴量1のデータ
        action_feat2(list):行動を取った時の特徴量2のデータ
        not_action_feat1(list):行動を取らない時の特徴量1のデータ
        not_action_feat2(list):行動を取らない時の特徴量2のデータ
    returns:
        feat1_a_set(list):特徴量1のデータ(行動を取るor取らないデータの2対)
        feat1_b_set(list):特徴量1のデータ(行動を取るor取らないデータの2対)
        feat2_a_set(list):特徴量2のデータ(行動を取るor取らないデータの2対)
        feat2_b_set(list):特徴量2のデータ(行動を取るor取らないデータの2対)
        labels(list):各データセットのラベル
    """

    def labeling_for_action(action_feat:list,not_action_feat:list):

        """
        ある特徴量の行動を取るor取らないデータの配列の全ての組み合わせに対して(1,0,-1)のラベルを付ける

        args:
            - action_feat(list):行動を取る場合の指定した特徴量データ
            - not_action_feat(list):行動を取らない場合の指定した特徴量データ
        return:
            - feat_a,feat_b,feat_y:特徴量データとラベル(1,0,-1)
        """

        #出力配列
        feat_a = []
        feat_b = []
        feat_y = []

        #全ての組み合わせに対してラベルを付ける
        for wave_1 in action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([1,1])
        for wave_1 in action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([1,0])

        for wave_1 in not_action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([0,1])

        for wave_1 in not_action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([0,0])

        return feat_a,feat_b,feat_y

    feat1_a,feat1_b,label = labeling_for_action(action_feat1,not_action_feat1)
    feat2_a,feat2_b,_ = labeling_for_action(action_feat2,not_action_feat2)


    # Combine the arrays into a list of tuples
    combined = list(zip(feat1_a, feat1_b, feat2_a, feat2_b, label))

    # Shuffle the list using random.shuffle()
    random.shuffle(combined)

    # Unpack the shuffled tuples back into separate arrays
    feat1_a_set, feat1_b_set, feat2_a_set, feat2_b_set, labels = zip(*combined)

    return feat1_a_set, feat1_b_set, feat2_a_set, feat2_b_set, labels

def normalization(data):

    # データの平均値と標準偏差を計算
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # data of normalization
    normalized_data = (data - mean) / std
    return normalized_data
"""
climo_walk_files = sorted([f for f in os.listdir('data/csv/climomaster') if 'walk' in f])
gauss_walk_files = sorted([f for f in os.listdir('data/csv/ML-logger') if 'walk' in f])

walk_wind_vel_list = []
walk_gauss_list = []
no_wind_vel_list = []
no_gauss_list = []

preprocess = preprocess_for_Siamese_Net()

for i, (climo_csv_path, gauss_csv_path) in enumerate(zip(climo_walk_files,gauss_walk_files)):
    
    #ファイルパスを指定する
    climo_walk_path = 'data/csv/climomaster/' + climo_csv_path
    gauss_walk_path = 'data/csv/ML-logger/' + gauss_csv_path
    climo_no_path = re.sub(r'-walk\d+', '', climo_walk_path)
    gauss_no_path = re.sub(r'-walk\d+', '', gauss_walk_path)

    #dfにする
    walk_merged_df = preprocess.convert_csv_to_mergedcsv(climo_walk_path,gauss_walk_path)
    no_merged_df = preprocess.convert_csv_to_mergedcsv(climo_no_path,gauss_no_path)

    walk_wind_vel,walk_gauss,no_wind_vel,no_gauss = generate_siamese_data(walk_merged_df,no_merged_df,4,120,300*(i+1))

    walk_wind_vel_list.extend(walk_wind_vel)
    walk_gauss_list.extend(walk_gauss)
    no_wind_vel_list.extend(no_wind_vel)
    no_gauss_list.extend(no_gauss)

    
wind_a_set,wind_b_set,gauss_a_set,gauss_b_set,labels = generate_npy_from_siamese_data(walk_wind_vel_list,
                                                                                      walk_gauss_list,
                                                                                      no_wind_vel_list,
                                                                                      no_gauss_list)

#npyファイルに変換
datadir = "data/train-npy/"

np.save(datadir + 'wind_a_set', wind_a_set)
np.save(datadir + 'wind_b_set', wind_b_set)
np.save(datadir + 'gauss_a_set', gauss_a_set)
np.save(datadir + 'gauss_b_set', gauss_b_set)
np.save(datadir + 'labels', labels)
"""

datadir = "data/train-npy/"
checkpoints_dir = "data/checkpoints/"
logs_dir = "data/logs/"

true_gauss = np.load(datadir + 'gauss_a_set.npy')
true_wind = np.load(datadir + 'wind_a_set.npy')
label = np.load(datadir + 'labels.npy')

n_max_gpus = torch.cuda.device_count()
print(f'{n_max_gpus} GPUs available')
n_gpus = min(2, n_max_gpus)
print(f'Using {n_gpus} GPUs')

train_data_len = 3000
val_data_len = 3500

#識別学習に用いるone-hot表現のラベルを作成
one_hot_labels = torch.zeros(val_data_len, 2, dtype=torch.float)
for step, genuine_label in enumerate(label[:val_data_len][:,0]):
    if genuine_label == 1:
        one_hot_labels[step]=torch.tensor([1,0],dtype=torch.float)
    if genuine_label == 0:
        one_hot_labels[step]=torch.tensor([0,1],dtype=torch.float)

true_gauss_normal = normalization(true_gauss[0:val_data_len])
true_wind_normal = normalization(true_wind[0:val_data_len])

traindataset = DummyDataset(true_gauss_normal[0:train_data_len],true_wind_normal[0:train_data_len],one_hot_labels[0:train_data_len])
valdataset = DummyDataset(true_gauss_normal[train_data_len:val_data_len],true_wind_normal[train_data_len:val_data_len],one_hot_labels[train_data_len:val_data_len])

epochs = 3
batch_size = 100
train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedEncoderCNN()
model.to(device)
lossfn = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

model.train()

torch.set_grad_enabled(True)
print("STARING TO TRAIN MODEL")
file1 = open(logs_dir + "CNN_classfier_train_accuracies.txt","w")
file2 = open(logs_dir + 'CNN_classfier_val_accuracies.txt','w')
for epoch in range(1, epochs+1):

    model.train()
    
    steps_losses = []
    steps_accu = []
    train_steps_accu = []
    val_steps_accu = []

    model_checkpoints = checkpoints_dir + "model_" + str(epoch) + ".pt"
    for steps, (true_gauss_tensor, true_wind_tensor, labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        optimizer.zero_grad() 

        genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
        #calculate contrastive loss
        loss = lossfn(genuine_output, labels.to(device))
        steps_losses.append(loss.cpu().detach().numpy())
        #calculate accuracy
        outputs_softmax = torch.softmax(genuine_output,dim=1)
        predicted_classes = torch.argmax(outputs_softmax, dim=1)
        # one-hot表現に変換
        predicted_labels = torch.zeros(genuine_output.size(0), 2)
        predicted_labels.scatter_(1, predicted_classes.cpu().unsqueeze(1), 1)
        correct = (predicted_labels.to(device) == labels.to(device)).sum().item()
        total = labels.numel()
        accuracy = correct / total
        train_steps_accu.append(accuracy)
        loss.backward()
        optimizer.step()

    now_time = dt.datetime.now()
    print(f"EPOCH {epoch}| Train: loss {np.mean(steps_losses)}| train accuracy {np.mean(train_steps_accu)} ")
    file1.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_losses)), "train_accuracy", str(np.mean(train_steps_accu)), now_time))
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for steps, (true_gauss_tensor, true_wind_tensor, labels) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):

            genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
            #calculate contrastive loss
            loss = lossfn(genuine_output, labels.to(device))
            steps_losses.append(loss.cpu().detach().numpy())
            #calculate accuracy
            outputs_softmax = torch.softmax(genuine_output,dim=1)
            predicted_classes = torch.argmax(outputs_softmax, dim=1)
            # one-hot表現に変換
            predicted_labels = torch.zeros(genuine_output.size(0), 2)
            predicted_labels.scatter_(1, predicted_classes.cpu().unsqueeze(1), 1)
            correct = (predicted_labels.to(device) == labels.to(device)).sum().item()
            total = labels.numel()
            accuracy = correct / total
            val_steps_accu.append(accuracy)

        print(f"EPOCH {epoch}| Val: loss {np.mean(steps_losses)}| val accuracy {np.mean(val_steps_accu)} ")
        file2.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "val_loss", str(np.mean(steps_losses)), "val_accuracy", str(np.mean(val_steps_accu)), now_time))
file1.close()
file2.close()