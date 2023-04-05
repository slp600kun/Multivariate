import numpy as np
from preprocess_data import preprocess_for_Siamese_Net
from train import ConvLayer2D,windEncoder,CombinedEncoder,DummyDataset,ContrastiveLoss
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
                                                                 not_action_df_starting_point+len(action_segment_data_list)],
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


test_walk_merged_df = preprocess.convert_csv_to_mergedcsv('data/csv/climomaster/2023-0318-walk.KAM.CSV','data/csv/ML-logger/2023-0318-walk-gauss.csv')
test_no_merged_df = preprocess.convert_csv_to_mergedcsv('data/csv/climomaster/2023-0318.KAM.CSV','data/csv/ML-logger/2023-0318-gauss.csv')

test_walk_wind_vel,test_walk_gauss,test_no_wind_vel,test_no_gauss = generate_siamese_data(test_walk_merged_df,test_no_merged_df,4,30,300)  

test_wind_a_set,test_wind_b_set,test_gauss_a_set,test_gauss_b_set,test_labels = generate_npy_from_siamese_data(test_walk_wind_vel,
                                                                                                               test_walk_gauss,
                                                                                                               test_no_wind_vel,
                                                                                                               test_no_gauss)
#npyファイルに変換
datadir = "data/train-npy/"
np.save(datadir + 'test_wind_a_set', test_wind_a_set)
np.save(datadir + 'test_wind_b_set', test_wind_b_set)
np.save(datadir + 'test_gauss_a_set', test_gauss_a_set)
np.save(datadir + 'test_gauss_b_set', test_gauss_b_set)
np.save(datadir + 'test_labels', test_labels)

logs_dir = "data/logs/"

#テストデータの学習
test_true_gauss = np.load(datadir + 'test_gauss_a_set.npy')
test_true_wind = np.load(datadir + 'test_wind_a_set.npy')
test_wrong_gauss = np.load(datadir + 'test_gauss_b_set.npy')
test_wrong_wind = np.load(datadir + 'test_wind_b_set.npy')
test_label = np.load(datadir + 'test_labels.npy')

batch_size = 128
testdataset = DummyDataset(test_true_gauss[0:30000],test_true_wind[0:30000],test_wrong_gauss[0:30000],
                       test_wrong_wind[0:30000],test_label[0:30000])
test_dataloader = DataLoader(testdataset, batch_size = batch_size, shuffle=False)
test_acc_list = []
file3 = open(logs_dir + 'test_accuracies.txt','w')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = ContrastiveLoss().to(device)
model = CombinedEncoder().to(device)

model_path = 'data/checkpoints/model_50.pt'
model.load_state_dict(torch.load(model_path))
model.eval()


with torch.no_grad():
    for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):

        true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 1)
        true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 1)
        wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 1)
        wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 1)
        true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 3)
        true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 3)
        wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 3)
        wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 3)

        genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
        forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))
        loss,y_pred = loss_fn(genuine_output, forged_output, labels.to(device))
        prediction = (y_pred.cpu().detach().numpy()>0.4).astype(np.int)
        accuracy = accuracy_score(labels,prediction)
        test_acc_list.append(accuracy)
        print(f"{steps}| test:  loss {np.mean(loss)}| accuracy {np.mean(accuracy)}")
