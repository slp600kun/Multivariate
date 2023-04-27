import numpy as np
import sys
import os
import re
from preprocess_data import preprocess_for_Siamese_Net
from train import ConvLayer2D,windEncoder,WindEncoderLSTM,CombinedEncoder,CombinedEncoderLSTM,DummyDataset,ContrastiveLoss
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import datetime as dt
import random
import pandas as pd

preprocess = preprocess_for_Siamese_Net()

def generate_siamese_data(action_df:pd,
                          not_action_df:pd,
                          slide_length:int,
                          segment_data_length:int,
                          not_action_df_starting_point:int,
                          is_wind_vel_converted: bool = True,
                          is_temp_converted: bool = False,
                          is_humid_converted: bool = False,
                          is_gauss_converted: bool = True):
    
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
    not_action_segment_data_list = slide_time_data(not_action_df[not_action_df_starting_point:
                                                                 not_action_df_starting_point+len(action_df)],
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
                feat_y.append(1)

        for wave_1 in not_action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(0)

        for wave_1 in action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(-1)

        for wave_1 in not_action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(-1)

        return feat_a,feat_b,feat_y

    feat1_a, feat1_b,feat1_y = labeling_for_action(action_feat1,not_action_feat1)
    feat2_a,feat2_b,feat2_y = labeling_for_action(action_feat2,not_action_feat2)

    label=[]

    #ラベリング
    for i in range(len(feat2_a)):
        if(feat1_y[i] == -1 or feat2_y[i] == -1):
            label.append(0)
        elif(feat1_y[i] == feat2_y[i]):
            label.append(1)
        else:
            label.append(0)

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

    # データの正規化
    normalized_data = (data - mean) / std
    return normalized_data

datadir = "data/train-npy/"
logs_dir = "data/logs/"

#テストデータ(一回分)
test_walk_merged_df = preprocess.convert_csv_to_mergedcsv('data/csv/climomaster/test/2023-0424-walk2.KAM.CSV','data/csv/ML-logger/test/2023-0424-walk2-gauss.csv')
test_no_merged_df = preprocess.convert_csv_to_mergedcsv('data/csv/climomaster/test/2023-0424.KAM.CSV','data/csv/ML-logger/test/2023-0424-gauss.csv')

test_walk_wind_vel,test_walk_gauss,test_no_wind_vel,test_no_gauss = generate_siamese_data(test_walk_merged_df,test_no_merged_df,4,60,300)  

test_wind_a_set,test_wind_b_set,test_gauss_a_set,test_gauss_b_set,test_labels = generate_npy_from_siamese_data(test_walk_wind_vel,
                                                                                                               test_walk_gauss,
                                                                                                               test_no_wind_vel,
                                                                                                               test_no_gauss)
#npyファイルに変換
np.save(datadir + 'test_wind_a_set', test_wind_a_set)
np.save(datadir + 'test_wind_b_set', test_wind_b_set)
np.save(datadir + 'test_gauss_a_set', test_gauss_a_set)
np.save(datadir + 'test_gauss_b_set', test_gauss_b_set)
np.save(datadir + 'test_labels', test_labels)

"""
climo_walk_files = sorted([f for f in os.listdir('data/csv/climomaster/test') if 'walk' in f])
gauss_walk_files = sorted([f for f in os.listdir('data/csv/ML-logger/test') if 'walk' in f])

walk_wind_vel_list = []
walk_gauss_list = []
no_wind_vel_list = []
no_gauss_list = []

preprocess = preprocess_for_Siamese_Net()

for i, (climo_csv_path, gauss_csv_path) in enumerate(zip(climo_walk_files,gauss_walk_files)):

    #ファイルパスを指定する
    climo_walk_path = 'data/csv/climomaster/test/' + climo_csv_path
    gauss_walk_path = 'data/csv/ML-logger/test/' + gauss_csv_path
    climo_no_path = re.sub(r'-walk\d+', '', climo_walk_path)
    gauss_no_path = re.sub(r'-walk\d+', '', gauss_walk_path)

    #dfにする
    walk_merged_df = preprocess.convert_csv_to_mergedcsv(climo_walk_path,gauss_walk_path)
    no_merged_df = preprocess.convert_csv_to_mergedcsv(climo_no_path,gauss_no_path)

    walk_wind_vel,walk_gauss,no_wind_vel,no_gauss = generate_siamese_data(walk_merged_df,no_merged_df,4,60,300*(i+1))

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

np.save(datadir + 'test_wind_a_set', wind_a_set)
np.save(datadir + 'test_wind_b_set', wind_b_set)
np.save(datadir + 'test_gauss_a_set', gauss_a_set)
np.save(datadir + 'test_gauss_b_set', gauss_b_set)
np.save(datadir + 'test_labels', labels)
"""

#テストデータ
test_true_gauss = np.load(datadir + 'test_gauss_a_set.npy')
test_true_wind = np.load(datadir + 'test_wind_a_set.npy')
test_wrong_gauss = np.load(datadir + 'test_gauss_b_set.npy')
test_wrong_wind = np.load(datadir + 'test_wind_b_set.npy')
test_label = np.load(datadir + 'test_labels.npy')

test_data_len = 45000
true_gauss_normal = normalization(test_true_gauss)
true_wind_normal = normalization(test_true_wind)
wrong_gauss_normal = normalization(test_wrong_gauss)
wrong_wind_normal = normalization(test_wrong_wind)

testdataset = DummyDataset(true_gauss_normal[0:test_data_len],true_wind_normal[0:test_data_len],wrong_gauss_normal[0:test_data_len],
                       wrong_wind_normal[0:test_data_len],test_label[0:test_data_len])
batch_size = 100
test_dataloader = DataLoader(testdataset, batch_size = batch_size, shuffle=True)

"""
#テストデータの学習(10回分)
test_true_gauss = np.load(datadir + 'gauss_a_set.npy')
test_true_wind = np.load(datadir + 'wind_a_set.npy')
test_wrong_gauss = np.load(datadir + 'gauss_b_set.npy')
test_wrong_wind = np.load(datadir + 'wind_b_set.npy')
test_label = np.load(datadir + 'labels.npy')

val_data_len = 80000
test_data_len = 90000

true_gauss_normal = normalization(test_true_gauss[0:test_data_len])
true_wind_normal = normalization(test_true_wind[0:test_data_len])
wrong_gauss_normal = normalization(test_wrong_gauss[0:test_data_len])
wrong_wind_normal = normalization(test_wrong_wind[0:test_data_len])

testdataset = DummyDataset(true_gauss_normal[val_data_len:test_data_len],true_wind_normal[val_data_len:test_data_len],wrong_gauss_normal[val_data_len:test_data_len],
                           wrong_wind_normal[val_data_len:test_data_len],test_label[val_data_len:test_data_len])
test_dataloader = DataLoader(testdataset, shuffle=False)
"""

test_acc_list = []
file3 = open(logs_dir + 'test_accuracies.txt','w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = ContrastiveLoss().to(device)
model = CombinedEncoderLSTM().to(device)

model_path = 'data/checkpoints/model_10.pt'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()

steps_accu = []
labels_array = []
predictions_array = []
with torch.no_grad():
    for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):

        true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 2)
        true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 2)
        wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 2)
        wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 2)
        #true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 3)
        #true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 3)
        #wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 3)
        #wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 3)

        genuine_output = model.forward(true_gauss_tensor.to(device), true_wind_tensor.to(device))
        forged_output = model.forward(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))

        #-1→1に変換(距離学習を行うため)
        abs_label = torch.abs(labels).int()
        loss,y_pred = loss_fn(genuine_output[0], forged_output[0], abs_label.to(device))
        prediction = (y_pred.cpu().detach().numpy()>0.4).astype(int)
        accuracy = accuracy_score(abs_label,prediction)
        steps_accu.append(accuracy)


def accuracy(y_pred, y_true):
    """正解率を計算する関数"""
    return np.mean(y_pred == y_true)

def precision(y_pred, y_true):
    """適合率を計算する関数"""
    tp = np.sum((y_pred == 1) & (y_true == 1))  # true positive
    fp = np.sum((y_pred == 1) & (y_true == 0))  # false positive
    return tp / (tp + fp)

def recall(y_pred, y_true):
    """再現率を計算する関数"""
    tp = np.sum((y_pred == 1) & (y_true == 1))  # true positive
    fn = np.sum((y_pred == 0) & (y_true == 1))  # false negative
    return tp / (tp + fn)

def f1_score(y_pred, y_true):
    """F値を計算する関数"""
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    return 2 * p * r / (p + r)


print(f"steps Accuracy: {np.mean(steps_accu)}")

acc = 0
prec = 0
rec = 0
f1 = 0
file3.write("%s,%s,\n%s,%s,\n%s,%s,\n%s,%s,\n" %("Accuracy",str(acc),"Precision",str(prec),"Recall",str(rec),"F1 score",str(f1)))
file3.close()
