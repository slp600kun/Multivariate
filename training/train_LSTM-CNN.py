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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.auto import tqdm
import datetime as dt
import random
import pandas as pd
torch.utils.backcompat.broadcast_warning.enabled = True

preprocess = preprocess_for_Siamese_Net()

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
    
class LSTM_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.LSTM_block_1 = nn.LSTM(input_size=1, hidden_size=128, num_layers=2,batch_first=True,dropout=0.2)
        self.LSTM_block_2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,batch_first=True,dropout=0.2)
        self.LSTM_block_3 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,batch_first=True,dropout=0.2)
        self.bn_block_1 = nn.BatchNorm1d(128)
        self.bn_block_2 = nn.BatchNorm1d(256)
        
        self.fc = nn.Linear(128,64)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

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
        out = self.fc(output)
        return out

class CombinedEncoderLSTM(nn.Module):
    def __init__(self):
        super(CombinedEncoderLSTM, self).__init__()
        self.gauss_enc = LSTM_embedding()
        self.wind_enc = LSTM_embedding()

    def forward(self, gauss_input, wind_input):
        gauss_output = self.gauss_enc(gauss_input)
        wind_output = self.wind_enc(wind_input)
        output = torch.cat((gauss_output,wind_output),dim=1)
        return output


class FullyConnected(nn.Module):
    def __init__(self):
        super().__init__()
        # Define parameters
        self.embedding = CombinedEncoderLSTM()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(False),
            nn.Linear(64,2),
        )

    def forward(self, gauss_input, wind_input):
        embedding_vector = self.embedding(gauss_input, wind_input)
        output = self.fc(embedding_vector)
        return output


class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.AdaptiveMaxPool1d(output_size=64),
            nn.Dropout1d(0.5),
            nn.Conv1d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.AdaptiveMaxPool1d(output_size=32),
            nn.Dropout1d(0.5),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveMaxPool1d(output_size=16),
            nn.Dropout1d(0.5),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveMaxPool1d(output_size=8),
            nn.Dropout1d(0.5)
        )
        self.encoder = CombinedEncoderLSTM()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024,64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, gauss_input,wind_input):
        x = self.encoder(gauss_input,wind_input)
        x = torch.unsqueeze(x, dim = 1)
        out = self.cnn(x)
        out = torch.flatten(out,start_dim=1)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels,device):
        # ベクトルの対応するラベルを取得
        anchor = embeddings[::2]
        positive = embeddings[1::2]
        label = (labels[::2] == labels[1::2]).float()
        label = [1 if item[0] == 1 and item[1] == 1 else 0 for item in label]

        # ベクトル間のユークリッド距離を計算
        distance = torch.norm(anchor - positive, dim=1)
        
        # Contrastive Lossを計算
        loss = torch.mean(torch.tensor(label, device=device) * distance.pow(2) + (1 - torch.tensor(label, device=device)) * torch.clamp(self.margin - distance, min=0).pow(2))
        return loss

class MLP(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128,num_classes)
        self.dropout1 = nn.Dropout2d(0.2)
        self.bn_block_1 = nn.BatchNorm1d(128)
        self.bn_block_2 = nn.BatchNorm1d(512)
        self.bn_block_3 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout2d(0.4)
        self.relu = nn.PReLU()
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.bn_block_1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.bn_block_2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout2(x)
        x = self.bn_block_3(x)
        x = self.fc4(x)
        return x
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
label = np.load(datadir + 'labels.npy')

n_max_gpus = torch.cuda.device_count()
print(f'{n_max_gpus} GPUs available')
n_gpus = min(2, n_max_gpus)
print(f'Using {n_gpus} GPUs')

train_data_len = 30000
val_data_len = 35000

#識別学習に用いるone-hot表現のラベルを作成
one_hot_labels = torch.zeros(val_data_len, 2, dtype=torch.float)
for step, genuine_label in enumerate(label[:val_data_len][:,0]):
    if genuine_label == 1:
        one_hot_labels[step]=torch.tensor([1,0],dtype=torch.float)
    if genuine_label == 0:
        one_hot_labels[step]=torch.tensor([0,1],dtype=torch.float)

scaler_gauss = StandardScaler()
scaler_wind = StandardScaler()

scaler_gauss.fit(true_gauss[0:train_data_len])
scaled_gauss = scaler_gauss.transform(true_gauss)

scaler_wind.fit(true_wind[0:train_data_len])
scaled_wind = scaler_wind.transform(true_wind)

traindataset = DummyDataset(scaled_gauss[0:train_data_len],scaled_wind[0:train_data_len],one_hot_labels[0:train_data_len])
valdataset = DummyDataset(scaled_gauss[train_data_len:val_data_len],scaled_wind[train_data_len:val_data_len],one_hot_labels[train_data_len:val_data_len])

epochs = 5
class_epochs = 30
batch_size = 1000
train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FullyConnected()
model.to(device)
lossfn = ContrastiveLoss().to(device)

cont_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=0.01)
cont_scheduler = torch.optim.lr_scheduler.StepLR(cont_optimizer, step_size=5, gamma=0.1, verbose=True)

model.train()

torch.set_grad_enabled(True)
print("STARING TO TRAIN MODEL")
file1 = open(logs_dir + "train_LSTM-CNN_accuracies.txt","w")
file2 = open(logs_dir + 'val_LSTM-CNN_accuracies.txt','w')

for epoch in range(1, epochs+1):

    model.train()
    #距離学習に必要な配列
    train_cont_losses = []
    val_cont_losses = []
    embedding_vector = []
 

    dist_model_checkpoints = checkpoints_dir + "dist_model_" + str(epoch) + ".pt"
    for steps, (train_gauss_tensor, train_wind_tensor, train_labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):        

        cont_optimizer.zero_grad() 
        train_genuine_output = model(train_gauss_tensor.to(device), train_wind_tensor.to(device))
        #calculate contrastive loss
        train_cont_loss = lossfn(train_genuine_output, train_labels.to(device),device)
        train_cont_losses.append(train_cont_loss.cpu().detach().numpy())        
        train_cont_loss.backward()
        cont_optimizer.step()

    print(f"EPOCH {epoch}| Train: loss {np.mean(train_cont_losses)}") 
    torch.save(model.state_dict(),dist_model_checkpoints)

    cont_scheduler.step()
    model.eval()
    with torch.no_grad():
        for steps, (val_gauss_tensor, val_wind_tensor, val_labels) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            
            val_genuine_output = model(val_gauss_tensor.to(device), val_wind_tensor.to(device))
            val_cont_loss = lossfn(val_genuine_output, val_labels.to(device),device)
            val_cont_losses.append(val_cont_loss.cpu().detach().numpy())
        print(f"EPOCH {epoch}| Val: dist loss {np.mean(val_cont_losses)}")

device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model_path = checkpoints_dir + "dist_model_" + str(epoch) + ".pt"
model.load_state_dict(torch.load(model_path))

# Create an instance of the CNN model
identify_model = CNN(num_classes=2)
identify_model.encoder = model.embedding
identify_model.to(device1)
# モデルの一部を凍結
for param in identify_model.encoder.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss().to(device1)
cross_en_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, identify_model.parameters()), lr=0.0001,weight_decay=0.01)
cross_en_scheduler = torch.optim.lr_scheduler.StepLR(cross_en_optimizer, step_size=10, gamma=0.1, verbose=True)

for epoch in range(1, class_epochs+1):
    identify_model.train()
    train_cross_en_losses = []
    val_cross_en_losses = []
    train_class_accuracies = []
    val_class_accuracies = []
    class_model_checkpoints = checkpoints_dir + "class_model_" + str(epoch) + ".pt"

    for steps, (train_gauss_tensor, train_wind_tensor, train_labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        cross_en_optimizer.zero_grad()
        # Forward pass
        train_outputs = identify_model(train_gauss_tensor.to(device1),train_wind_tensor.to(device1))
        #calculate cross entropy loss
        _, train_y_targets = train_labels.clone().max(dim=1)
        train_cross_en_loss = criterion(train_outputs.requires_grad_(True), train_y_targets.long().to(device1))
        train_cross_en_losses.append(train_cross_en_loss.cpu().detach().numpy())
        #calculate accuracy
        train_outputs_softmax = torch.softmax(train_outputs,dim=1)
        train_predicted_classes = torch.argmax(train_outputs_softmax, dim=1)
        # one-hot表現に変換
        train_predicted_labels = torch.zeros(train_outputs.size(0), 2)
        train_predicted_labels.scatter_(1, train_predicted_classes.cpu().unsqueeze(1), 1)
        train_correct = (train_predicted_labels.to(device1) == train_labels.to(device1)).sum().item()
        train_total = train_labels.numel()
        train_accuracy = train_correct / train_total
        train_class_accuracies.append(train_accuracy)
        
        train_cross_en_loss.backward()
        cross_en_optimizer.step()

    now_time = dt.datetime.now()
    print(f"EPOCH {epoch}| Train: class loss  {np.mean(train_cross_en_losses)}| train accuracy {np.mean(train_class_accuracies)} ")
    file1.write("%s, %s, %s, %s, %s, %s\n" % (str(epoch), "train_cross_en_loss", str(np.mean(train_cross_en_losses)), "train_accuracy", str(np.mean(train_class_accuracies)), now_time))
    torch.save(identify_model.state_dict(),class_model_checkpoints)
    cross_en_scheduler.step()
    identify_model.eval()
    with torch.no_grad():
        for steps, (val_gauss_tensor, val_wind_tensor, val_labels) in tqdm(enumerate(val_dataloader),total=len(val_dataloader)):
            # Forward pass
            val_outputs = identify_model(val_gauss_tensor.to(device1), val_wind_tensor.to(device1))
            #calculate loss
            _, y_val_targets = val_labels.clone().max(dim=1)
            val_cross_en_loss = criterion(val_outputs.requires_grad_(True), y_val_targets.long().to(device1))
            val_cross_en_losses.append(val_cross_en_loss.cpu().detach().numpy())
            #calculate accuracy
            val_outputs_softmax = torch.softmax(val_outputs,dim=1)
            val_predicted_classes = torch.argmax(val_outputs_softmax, dim=1)
            # one-hot表現に変換
            val_predicted_labels = torch.zeros(val_outputs.size(0), 2)
            val_predicted_labels.scatter_(1, val_predicted_classes.cpu().unsqueeze(1), 1)
            val_correct = (val_predicted_labels.to(device1) == val_labels.to(device1)).sum().item()
            val_total = val_labels.numel()
            val_accuracy = val_correct / val_total
            val_class_accuracies.append(val_accuracy)

        print(f"EPOCH {epoch}| Val: class loss  {np.mean(val_cross_en_losses)} | val accuracy {np.mean(val_class_accuracies)} ")
        file2.write("%s, %s, %s, %s, %s, %s\n" % (str(epoch), "val_cross_en_loss", str(np.mean(val_cross_en_losses)), "val_accuracy", str(np.mean(val_class_accuracies)), now_time))
file1.close()
file2.close()

datadir = "data/train-npy/"
logs_dir = "data/logs/"

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
test_label = np.load(datadir + 'test_labels.npy')

test_data_len = 50000
#識別学習に用いるone-hot表現のラベルを作成
one_hot_testlabels = torch.zeros(test_data_len, 2, dtype=torch.float)
for step, genuine_label in enumerate(test_label[:test_data_len][:,0]):
    if genuine_label == 1:
        one_hot_testlabels[step]=torch.tensor([1,0],dtype=torch.float)
    if genuine_label == 0:
        one_hot_testlabels[step]=torch.tensor([0,1],dtype=torch.float)
#スケーリング
test_scaled_gauss = scaler_gauss.transform(test_true_gauss)
test_scaled_wind = scaler_wind.transform(test_true_wind)
#データ読み込み
testdataset = DummyDataset(test_scaled_gauss[0:test_data_len] ,test_scaled_wind[0:test_data_len] ,one_hot_testlabels[0:test_data_len])
batch_size = 100
test_dataloader = DataLoader(testdataset, batch_size = batch_size, shuffle=True)
#モデル構築
class_model_path = checkpoints_dir + "class_model_" + str(class_epochs) + ".pt"
class_checkpoint = torch.load(class_model_path)
model.load_state_dict(torch.load(model_path))
identify_model.load_state_dict(class_checkpoint)
identify_model.encoder = model.embedding
# モデルの一部を凍結
for param in identify_model.encoder.parameters():
    param.requires_grad = False

identify_model.eval()
#統計指標
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []
#テスト
with torch.no_grad():
    for steps, (test_gauss_tensor, test_wind_tensor, test_labels) in tqdm(enumerate(test_dataloader),total=len(test_dataloader)):
        test_outputs = identify_model(test_gauss_tensor.to(device1), test_wind_tensor.to(device1))
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
file3 = open(logs_dir + "test_LSTM-CNN_metrics.txt","w")
file3.write("%s,%s,\n%s,%s,\n%s,%s,\n%s,%s,\n" %("Accuracy",str(acc),"Precision",str(prec),"Recall",str(recall),"F1 score",str(f1)))
file3.close()
