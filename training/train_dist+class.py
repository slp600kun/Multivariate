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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import datetime as dt
import random
import pandas as pd

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
        self.dense_1 = nn.Linear(128, 64)
        self.dense_2 = nn.Linear(64, 1)
        self.dense_3 = nn.Linear(64, 32)
        self.bn_block_4 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(False)
        self.dropout_1 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.gauss_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        #embedding
        embedding_vector = output
        
        output = self.dense_1(output)
        output = self.relu(output)
        output = self.dropout_1(output)
        output = self.bn_block_4(output)

        #距離学習に対する出力
        output_one_dim = self.dense_2(output)
        output_one_dim = self.sigmoid(output_one_dim)
        
        #SVM
        #output_two_dim = self.dense_3(output)

        return output_one_dim, embedding_vector

class LSTM_embedding(nn.Module):
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

        output,_ = self.LSTM_block_3(output)
        output = output.permute(0, 2, 1)
        output = self.bn_block_3(output)
        output = output.permute(0, 2, 1)
        output = output.reshape(output.size(0), -1)
        return output
    

class CombinedEncoderLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.gauss_enc = LSTM_embedding()
        self.wind_enc = LSTM_embedding()
        self.dense_1 = nn.Linear(7680, 60)
        self.dense_2 = nn.Linear(60, 2)
        self.dense_3 = nn.Linear(60, 32)
        self.bn_block_4 = nn.BatchNorm1d(60)
        self.relu = nn.ReLU(False)
        self.dropout_1 = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.gauss_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),1)
        all_output = output
        
        output = self.dense_1(output)
        output = self.relu(output)
        output = self.dropout_1(output)
        output = self.bn_block_4(output)

        #距離学習に対する出力
        output_two_dim = self.dense_2(output)
    
        return output_two_dim,all_output

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
        return hamming_loss, y_pred_sign

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
wrong_gauss = np.load(datadir + 'gauss_b_set.npy')
wrong_wind = np.load(datadir + 'wind_b_set.npy')
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
wrong_gauss_normal = normalization(wrong_gauss[0:val_data_len])
wrong_wind_normal = normalization(wrong_wind[0:val_data_len])

traindataset = DummyDataset(true_gauss_normal[0:train_data_len],true_wind_normal[0:train_data_len],wrong_gauss_normal[0:train_data_len],
                       wrong_wind_normal[0:train_data_len],label[0:train_data_len])

valdataset = DummyDataset(true_gauss_normal[train_data_len:val_data_len],true_wind_normal[train_data_len:val_data_len],
                        wrong_gauss_normal[train_data_len:val_data_len],wrong_wind_normal[train_data_len:val_data_len],label[train_data_len:val_data_len])

epochs = 1
batch_size = 100
train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loss_fn = nn.CosineEmbeddingLoss().to(device)
model = CombinedEncoderLSTM()
#model = nn.DataParallel(model)
model.to(device)
contrastive_lossfn = ContrastiveLoss().to(device)
classifier_lossfn = HammingLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
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
    feature_vector_val = []
    
    steps_const_losses = []
    steps_identifical_accu = []

    model_checkpoints = checkpoints_dir + "model_" + str(epoch) + ".pt"
    for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        optimizer.zero_grad()
        
        #generate identical or not identical labels
        identical_labels = torch.zeros(batch_size)
        for i in range(batch_size):
            if labels[i, 0] == labels[i, 1]:
                identical_labels[i] = 1
            else:
                identical_labels[i] = 0
        
        genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
        forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))
        #calculate contrastive loss
        feature_vector_train.append(genuine_output[1])
        contrastive_loss,y_contrastive_pred = contrastive_lossfn(genuine_output[0], forged_output[0], identical_labels.to(device))

        steps_const_losses.append(contrastive_loss.cpu().detach().numpy())
        identifical_prediction = (y_contrastive_pred.cpu().detach().numpy()>0.4).astype(int)
        identifical_accuracy = accuracy_score(identical_labels,identifical_prediction)
        steps_identifical_accu.append(identifical_accuracy)
        contrastive_loss.backward()
        optimizer.step()

    embedding_vector_train = torch.cat(feature_vector_train,dim=0)
    print(embedding_vector_train.shape)

    now_time = dt.datetime.now()
    print(f"EPOCH {epoch}| Train: contrastive loss {np.mean(steps_const_losses)}| identifical accuracy {np.mean(steps_identifical_accu)} ")
    const_file1.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "train_loss", str(np.mean(steps_const_losses)), "train_accuracy", str(np.mean(steps_identifical_accu)), now_time))
    scheduler.step()
    model.eval()
    with torch.no_grad():
        for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):

            genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
            forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))
            #calculate contrastive loss
            contrastive_loss,y_contrastive_pred = contrastive_lossfn(genuine_output[0], forged_output[0], identical_labels.to(device))
            
            feature_vector_val.append(genuine_output[1])
            steps_const_losses.append(contrastive_loss.cpu().detach().numpy())
            identifical_prediction = (y_contrastive_pred.cpu().detach().numpy()>0.4).astype(int)
            identifical_accuracy = accuracy_score(identical_labels,identifical_prediction)
            steps_identifical_accu.append(identifical_accuracy)

        embedding_vector_val = torch.cat(feature_vector_val,dim=0)

        print(f"EPOCH {epoch}| Val: contrastive loss {np.mean(steps_const_losses)}| identifical accuracy {np.mean(steps_identifical_accu)} ")
        const_file2.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "val_loss", str(np.mean(steps_const_losses)), "val_accuracy", str(np.mean(steps_identifical_accu)), now_time))
const_file1.close()
const_file2.close()


# Define the MLP model
class MLP(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.fc1 = nn.Linear(7680, 128)
        #self.fc2 = nn.Linear(256, 512)
        #self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(128,num_classes)
        self.dropout1 = nn.Dropout2d(0.2)
        self.bn_block_1 = nn.BatchNorm1d(128)
        #self.bn_block_2 = nn.BatchNorm1d(512)
        #self.bn_block_3 = nn.BatchNorm1d(64)
        #self.dropout2 = nn.Dropout2d(0.2)
        self.relu = nn.PReLU()
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.bn_block_1(x)
        """
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.bn_block_2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.bn_block_3(x)
        """
        x = self.fc4(x)
        return x

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(128, 256, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=1, stride=1)
        self.conv3 = nn.Conv1d(512, 1024, kernel_size=1, stride=1)
        self.conv4 = nn.Conv1d(1024, 4096, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool1 = nn.AdaptiveMaxPool1d(output_size=16)
        self.maxpool2 = nn.AdaptiveMaxPool1d(output_size=8)
        self.maxpool3 = nn.AdaptiveMaxPool1d(output_size=4)
        self.maxpool4 = nn.AdaptiveMaxPool1d(output_size=1)
        self.norm1 = nn.BatchNorm1d(256)
        self.norm2 = nn.BatchNorm1d(512)
        self.norm3 = nn.BatchNorm1d(1024)
        self.norm4 = nn.BatchNorm1d(4096)
        self.drop = nn.Dropout1d(0.4)
        self.fc1 = nn.Linear(4096,1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, dim = 2)
        out = self.conv1(x) 
        out = self.relu(out)
        out = self.norm1(out)
        out = self.maxpool1(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.maxpool2(out)
        out = self.drop(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.norm3(out)
        out = self.maxpool3(out)
        out = self.drop(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.norm4(out)
        out = self.maxpool4(out)
        out = self.drop(out)

        out = torch.flatten(out, 1) 
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Create an instance of the MLP model
identify_model = MLP(num_classes=2)
identify_model = nn.DataParallel(identify_model,device_ids=[1])
identify_model.to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
identify_optimizer = torch.optim.SGD(identify_model.parameters(), lr=0.1,momentum=0.9,weight_decay=0.0001)
#identify_scheduler = torch.optim.lr_scheduler.StepLR(identify_optimizer, step_size=100, gamma=0.1)

train_identify_len = int(train_data_len*0.8)

min_value = embedding_vector_train[0:train_data_len].min()
max_value = embedding_vector_train[0:train_data_len].max()
normalized_embedding_vector = (embedding_vector_train[0:train_data_len] - min_value) / (max_value - min_value)

identify_train_dataset = TensorDataset(normalized_embedding_vector[0:train_identify_len],one_hot_labels[0:train_identify_len])
identify_val_dataset = TensorDataset(normalized_embedding_vector[train_identify_len:train_data_len],one_hot_labels[train_identify_len:train_data_len])

identify_train_dataloader = DataLoader(identify_train_dataset, batch_size=100, shuffle=True)
identify_val_dataloader = DataLoader(identify_val_dataset, batch_size=100, shuffle=True)

# Train the model for 50 epochs
num_epochs = 1000
for epoch in range(num_epochs):
    identify_model.train()

    train_steps_losses = []
    train_steps_accuracies = []
    val_steps_losses = []
    val_steps_accuracies = []

    for steps, (train_embedding_vector, train_labels) in tqdm(enumerate(identify_train_dataloader),total=len(identify_train_dataloader)):

        # Forward pass
        train_embedding_vector_copy = torch.from_numpy(train_embedding_vector.detach().cpu().numpy())
        outputs = identify_model(train_embedding_vector_copy.to(device))
        #calculate loss
        _, y_targets = train_labels.clone().max(dim=1)
        loss = criterion(outputs.requires_grad_(True), y_targets.long().to(device))
        train_steps_losses.append(loss.cpu().detach().numpy())
        #calculate accuracy
        outputs_softmax = torch.softmax(outputs,dim=1)
        predicted_classes = torch.argmax(outputs_softmax, dim=1)
        # one-hot表現に変換
        predicted_labels = torch.zeros(outputs.size(0), 2)
        predicted_labels.scatter_(1, predicted_classes.cpu().unsqueeze(1), 1)
        correct = (predicted_labels.to(device) == train_labels.to(device)).sum().item()
        total = train_labels.numel()
        accuracy = correct / total
        train_steps_accuracies.append(accuracy)

        identify_optimizer.zero_grad()
        # Backward pass and optimization
        loss.backward()
        identify_optimizer.step()

    mean_loss = np.mean(train_steps_losses)
    mean_accuracy = np.mean(train_steps_accuracies)
    print(f"epoch {epoch}| train loss:{mean_loss} | train accuracy:{mean_accuracy}")
    
    #identify_scheduler.step()
    identify_model.eval()
    #with torch.no_grad():
    for steps,(val_embedding_vector, val_labels) in tqdm(enumerate(identify_val_dataloader),total=len(identify_val_dataloader)):

            # Forward pass
            val_embedding_vector_copy = torch.from_numpy(val_embedding_vector.detach().cpu().numpy())
            val_outputs = identify_model(val_embedding_vector_copy.to(device))
            #calculate loss
            _, y_val_targets = val_labels.clone().max(dim=1)
            val_loss = criterion(val_outputs, y_val_targets.long().to(device))
            val_steps_losses.append(val_loss.cpu().detach().numpy())
            #calculate accuracy
            val_outputs_softmax = torch.softmax(val_outputs,dim=1)
            val_predicted_classes = torch.argmax(val_outputs_softmax, dim=1)
            # one-hot表現に変換
            val_predicted_labels = torch.zeros(val_outputs.size(0), 2)
            val_predicted_labels.scatter_(1, val_predicted_classes.cpu().unsqueeze(1), 1)
            val_correct = (val_predicted_labels.to(device) == val_labels.to(device)).sum().item()
            val_total = val_labels.numel()
            val_accuracy = val_correct / val_total
            val_steps_accuracies.append(val_accuracy)
    
    val_mean_loss = np.mean(val_steps_losses)
    val_mean_accuracy = np.mean(val_steps_accuracies)
    print(f"epoch {epoch}| val loss:{val_mean_loss} | val accuracy:{val_mean_accuracy}")

""" 
    with torch.no_grad():
        for steps, (val_embedding_vector, val_labels) in tqdm(enumerate(identify_val_dataloader),total=len(identify_val_dataloader)):
            val_embedding_vector = torch.unsqueeze(val_embedding_vector, dim = 2)
            val_embedding_vector = torch.unsqueeze(val_embedding_vector, dim = 1)
            
            # Forward pass
            outputs = identify_model(val_embedding_vector.to(device))
            outputs = outputs.cpu().detach().requires_grad_(True)
            _, y_targets = train_labels.max(dim=1)
            loss = criterion(outputs, y_targets.long())
            val_loss += loss.item()
        mean_loss = val_loss / steps
        print(f"epoch {epoch}| train loss:{mean_loss}")
"""

def match_rate(arr1, arr2):
    count = 0
    for x, y in zip(arr1, arr2):
        if x == y:
            count += 1
    return count / len(arr1)

