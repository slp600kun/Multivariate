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
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import datetime as dt
import random
import pandas as pd

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

        return output_one_dim

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
        loss = torch.sum(loss) / y.size()[0]
        return loss, mdist
    
class HammingLoss(torch.nn.Module):

    def __init__(self,threshold=0.4):
        super().__init__()
        self.threshold = threshold

    def forward(self,x,y):
        multi_label_pred_list = []
        total_hamming_loss = 0
        y_pred_sign = (x >= self.threshold).float()
        hamming_loss = torch.mean(torch.abs(y_pred_sign - y))
        total_hamming_loss += hamming_loss
        multi_label_pred_list.append(y_pred_sign)
        multi_label = torch.cat(multi_label_pred_list, dim=1)
        mean_hamming_loss = total_hamming_loss / labels.shape[1]
        return mean_hamming_loss,multi_label

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
                feat_y.append(1)

        for wave_1 in not_action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(0)

        return feat_a,feat_b,feat_y

    feat1_a,feat1_b,feat1_y = labeling_for_action(action_feat1,not_action_feat1)
    feat2_a,feat2_b,feat2_y = labeling_for_action(action_feat2,not_action_feat2)

    label=[]

    #labeling
    for i in range(len(feat2_a)):
        if(feat1_y[i] == 1 or feat2_y[i] == 1):
            label.append([1,1])
        elif(feat1_y[i] == 1 or feat2_y[i] == 0):
            label.append([1,0])
        elif(feat1_y[i] == 0 or feat2_y[i] == 1):
            label.append([0,1])
        elif(feat1_y[i] == 0 or feat2_y[i] == 0):
            label.append([0,0])

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
wrong_gauss = np.load(datadir + 'gauss_b_set.npy')
wrong_wind = np.load(datadir + 'wind_b_set.npy')
label = np.load(datadir + 'labels.npy')

train_data_len = 30000
val_data_len = 35000

true_gauss_normal = normalization(true_gauss[0:val_data_len])
true_wind_normal = normalization(true_wind[0:val_data_len])
wrong_gauss_normal = normalization(wrong_gauss[0:val_data_len])
wrong_wind_normal = normalization(wrong_wind[0:val_data_len])

traindataset = DummyDataset(true_gauss_normal[0:train_data_len],true_wind_normal[0:train_data_len],wrong_gauss_normal[0:train_data_len],
                       wrong_wind_normal[0:train_data_len],label[0:train_data_len])

valdataset = DummyDataset(true_gauss_normal[train_data_len:val_data_len],true_wind_normal[train_data_len:val_data_len],
                        wrong_gauss_normal[train_data_len:val_data_len],wrong_wind_normal[train_data_len:val_data_len],label[train_data_len:val_data_len])

epochs = 50
batch_size = 100
train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loss_fn = nn.CosineEmbeddingLoss().to(device)
contrastive_lossfn = ContrastiveLoss().to(device)
classifier_lossfn = HammingLoss().to(device)
model = CombinedEncoderLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

model.train()

torch.set_grad_enabled(True)
print("STARING TO TRAIN MODEL")
const_file1 = open(logs_dir + "train_const_accuracies.txt","w")
classifier_file1 = open(logs_dir + "train_classifier_accuracies.txt","w")
const_file2 = open(logs_dir + 'validation_const_accuracies.txt','w')
classifier_file2 = open(logs_dir + 'validation_classifier_accuracies.txt','w')
file3 = open(logs_dir + 'svm_accuracies.txt' ,'w')
for epoch in range(1, epochs+1):

    model.train()
    #svm学習に必要な配列
    feature_vector_train = []
    feature_label_train = []
    feature_vector_val = []
    feature_label_val = []
    
    steps_const_losses = []
    steps_identifical_accu = []
    steps_hamming_losses = []
    steps_classifier_accu = []


    model_checkpoints = checkpoints_dir + "model_" + str(epoch) + ".pt"
    for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        optimizer.zero_grad() 
        true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 2)
        true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 2)
        wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 2)
        wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 2)
        #true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 3)
        #true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 3)
        #wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 3)
        #wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 3)
        
        #generate identical or not identical labels
        identical_labels = torch.zeros(100)
        for i in range(100):
            if labels[i, 0] == labels[i, 1]:
                identical_labels[i] = 1
            else:
                identical_labels[i] = 0

        genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
        forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))
        #calculate contrastive loss
        contrastive_loss,y_contrastive_pred = contrastive_lossfn(genuine_output, forged_output, identical_labels.to(device))

        #calculate genuine hamming loss
        genuine_mean_hamming_loss,genuine_multi_label = classifier_lossfn(genuine_output, labels[:,0])
        forged_mean_hamming_loss,forged_multi_label = classifier_lossfn(forged_output, labels[:,1])
        pred_labels = torch.cat([genuine_multi_label,forged_multi_label],dim=1)
        hamming_loss = (genuine_mean_hamming_loss + forged_mean_hamming_loss) / 2

        steps_const_losses.append(contrastive_loss.cpu().detach().numpy())
        identifical_prediction = (y_contrastive_pred.cpu().detach().numpy()>0.4).astype(int)
        identifical_accuracy = accuracy_score(identical_labels,identifical_prediction)
        steps_identifical_accu.append(identifical_accuracy)

        steps_hamming_losses.append(hamming_loss.cpu().detach().numpy())
        classifier_accuracy_0 = accuracy_score(labels[:,0],pred_labels[:,0])
        classifier_accuracy_1 = accuracy_score(labels[:,1],pred_labels[:,1])
        steps_classifier_accu.append((classifier_accuracy_0 + classifier_accuracy_1)/2)

        contrastive_loss.backward()
        optimizer.step()

    now_time = dt.datetime.now()
    print(f"EPOCH {epoch}| Train: contrastive loss {np.mean(steps_const_losses)}| identifical accuracy {np.mean(steps_identifical_accu)} ")
    print(f"EPOCH {epoch}| Train: hamming loss {np.mean(steps_hamming_losses)}| classifier accuracy {np.mean(steps_classifier_accu)} ")
    const_file1.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_const_losses)), "train_accuracy", str(np.mean(steps_identifical_accu)), now_time))
    classifier_file1.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_hamming_losses)), "train_accuracy", str(np.mean(steps_classifier_accu)), now_time))
    torch.save(model.state_dict(),model_checkpoints)
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 2)
            true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 2)
            wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 2)
            wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 2)
            #true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 3)
            #true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 3)
            #wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 3)
            #wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 3)
            #calculate contrastive loss
            contrastive_loss,y_contrastive_pred = contrastive_lossfn(genuine_output, forged_output, identical_labels.to(device))

            #calculate genuine hamming loss
            genuine_mean_hamming_loss,genuine_multi_label = classifier_lossfn(genuine_output, labels[:,0])
            forged_mean_hamming_loss,forged_multi_label = classifier_lossfn(forged_output, labels[:,1])
            pred_labels = torch.cat([genuine_multi_label,forged_multi_label],dim=1)
            hamming_loss = (genuine_mean_hamming_loss + forged_mean_hamming_loss) / 2

            steps_const_losses.append(contrastive_loss.cpu().detach().numpy())
            identifical_prediction = (y_contrastive_pred.cpu().detach().numpy()>0.4).astype(int)
            identifical_accuracy = accuracy_score(identical_labels,identifical_prediction)
            steps_identifical_accu.append(identifical_accuracy)

            steps_hamming_losses.append(hamming_loss.cpu().detach().numpy())
            classifier_accuracy_0 = accuracy_score(labels[:,0],pred_labels[:,0])
            classifier_accuracy_1 = accuracy_score(labels[:,1],pred_labels[:,1])
            steps_classifier_accu.append((classifier_accuracy_0 + classifier_accuracy_1)/2)

            contrastive_loss.backward()
            optimizer.step()        
        
        print(f"EPOCH {epoch}| Train: contrastive loss {np.mean(steps_const_losses)}| identifical accuracy {np.mean(steps_identifical_accu)} ")
        print(f"EPOCH {epoch}| Train: hamming loss {np.mean(steps_hamming_losses)}| classifier accuracy {np.mean(steps_classifier_accu)} ")
        const_file2.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_const_losses)), "train_accuracy", str(np.mean(steps_identifical_accu)), now_time))
        classifier_file2.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_hamming_losses)), "train_accuracy", str(np.mean(steps_classifier_accu)), now_time))

const_file1.close()
classifier_file1.close()

const_file2.close()
classifier_file2.close()

