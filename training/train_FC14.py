import sys
import os
import re
import itertools
from sklearn.manifold import TSNE
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
        
def generate_siamese_data(action_df:pd,
                          slide_length:int,
                          segment_data_length:int):
    
    """
    dfの対から必要なデータ列を選択し、一定の長さで
    """

    def slide_time_data(df:pd,slide_length:int,segment_data_length:int,
                            is_wind_converted: bool = True,
                            is_temp_converted: bool = False,
                            is_humid_converted: bool = False,
                            is_gauss_converted: bool = True) ->list:
        """
        merged_dfから必要なデータ列を選択し、一定の長さの窓を指定した秒間隔でスライドさせた状態で出力する
        args: 
            - df (pd): 元のmerged_df
            - slide_length(int): スライド間隔(秒)
            - segment_data_length (int):データの分割窓長さ(秒)
            - is _wind_converted (bool) :風速を出力に入れるか
            - is _temp_converted (bool) :温度を出力に入れるか
            - is _humid_converted (bool) :湿度を出力に入れるか
            - is _gauss_converted (bool) :磁束密度を出力に入れるか

        return:
            - segmented_datafarme_array(list)選択したデータをスライド出力したもの
        """ 
        
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

    #分割されたdf配列(この状態だとデータフレームの配列になる)
    action_segment_data_list = slide_time_data(action_df,slide_length,segment_data_length)

    #df先頭の特徴量のリスト
    feat_list = list(action_segment_data_list[0].columns.values.tolist())

    #各配列のdfを配列に変換
    for action_df in action_segment_data_list:
        action_feat1.append(action_df[feat_list[0]].values)
        action_feat2.append(action_df[feat_list[1]].values)
    return action_feat1,action_feat2

def process_files(data_type):

    def extract_number(filename,keyword):
        # ファイル名からキーワードの後の数字を抽出
        match = re.search(f'{keyword}(\d+)', filename)
        if match:
            return int(match.group(1))
        return float('inf')  # マッチしない場合は無限大を返す

    climo_files = sorted([filename for filename in os.listdir(f'data/csv/climomaster/{data_type}') if filename.endswith(".CSV")], 
                            key=lambda x: extract_number(x, "KAM"))
    gauss_files = sorted([filename for filename in os.listdir(f'data/csv/ML-logger/{data_type}')],
                            key=lambda x: extract_number(x, data_type))
    mvr_files = sorted([filename for filename in os.listdir(f'data/csv/MVR-RF10/{data_type}')],
                            key=lambda x: extract_number(x, data_type))

    all_dfs = []

    for climo_csv_path, gauss_csv_path ,mvr_csv_path in zip(climo_files, gauss_files, mvr_files):
        # ファイルパスを指定する
        climo_path = f'data/csv/climomaster/{data_type}/' + climo_csv_path
        gauss_path = f'data/csv/ML-logger/{data_type}/' + gauss_csv_path
        mvr_path = f'data/csv/MVR-RF10/{data_type}/' + mvr_csv_path
        # dfにする
        climo_gauss_df = preprocess.convert_csv_to_mergedcsv(climo_path, gauss_path)
        vib_df = preprocess.vibration_csv_to_df(mvr_path)
        filtered_climo_gauss_df,filtered_vib_df = preprocess.filter_data_by_time(climo_gauss_df,vib_df)
        
        all_dfs.append((filtered_climo_gauss_df,filtered_vib_df))

    return all_dfs

def generate_npy_from_siamese_data(action_feat1:list,action_feat2:list,not_action_feat1:list,not_action_feat2:list,additional_action_feat1:list,additional_action_feat2:list):

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

    def labeling_for_action(action_feat: list, not_action_feat: list, additional_action_feat: list):
        
        """
        ある特徴量の行動を取るor取らないデータの配列の全ての組み合わせに対してラベル(0,1)を付ける

        args:
            - action_feat(list): 行動を取る場合の指定した特徴量データ
            - not_action_feat(list): 行動を取らない場合の指定した特徴量データ
            - additional_action_feat(list): 追加の行動を取る場合の指定した特徴量データ
        return:
            - feat_a, feat_b, feat_y: 特徴量データとラベル
        """

        feat_a = []
        feat_b = []
        feat_y = []

        for wave_1 in not_action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([0,0])
        for wave_1 in not_action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([0,1])
        for wave_1 in not_action_feat:
            for wave_2 in additional_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([0,2])

        for wave_1 in action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([1,0])
        for wave_1 in action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([1,1])
        for wave_1 in action_feat:
            for wave_2 in additional_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([1,2])

        for wave_1 in additional_action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([2,0])
        for wave_1 in additional_action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([2,1])
        for wave_1 in additional_action_feat:
            for wave_2 in additional_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append([2,2])

        return feat_a, feat_b, feat_y
    
    feat1_a,feat1_b,label = labeling_for_action(action_feat1,not_action_feat1,additional_action_feat1)
    feat2_a,feat2_b,_ = labeling_for_action(action_feat2,not_action_feat2,additional_action_feat2)

    # Combine the arrays into a list of tuples
    combined = list(zip(feat1_a, feat1_b, feat2_a, feat2_b, label))

    # Shuffle the list using random.shuffle()
    random.shuffle(combined)

    # Unpack the shuffled tuples back into separate arrays
    feat1_a_set, feat1_b_set, feat2_a_set, feat2_b_set, labels = zip(*combined)

    return feat1_a_set, feat1_b_set, feat2_a_set, feat2_b_set, labels

def generate_npy_from_discriminate(action_feat1:list,action_feat2:list,not_action_feat1:list,not_action_feat2:list,additional_action_feat1:list,additional_action_feat2:list):

    """
    discriminate dataをラベリングし、npyファイルに出力する関数

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

    def labeling_for_action(action_feat: list, not_action_feat: list, additional_action_feat: list):
        
        """
        ある特徴量の行動を取るor取らないデータの配列の全ての組み合わせに対してラベル(0,1)を付ける

        args:
            - action_feat(list): 行動を取る場合の指定した特徴量データ
            - not_action_feat(list): 行動を取らない場合の指定した特徴量データ
            - additional_action_feat(list): 追加の行動を取る場合の指定した特徴量データ
        return:
            - feat_a, feat_b, feat_y: 特徴量データとラベル
        """

        feat_a = []
        feat_b = []
        feat_y = []

        for wave_1 in not_action_feat:
            for wave_2 in not_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(0)

        for wave_1 in action_feat:
            for wave_2 in action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(1)

        for wave_1 in additional_action_feat:
            for wave_2 in additional_action_feat:
                feat_a.append(wave_1)
                feat_b.append(wave_2)
                feat_y.append(2)

        return feat_a, feat_b, feat_y
    
    feat1_a,feat1_b,label = labeling_for_action(action_feat1[:200],not_action_feat1[:1000],additional_action_feat1[:200])

    # Combine the arrays into a list of tuples
    combined = list(zip(feat1_a, feat1_b, label))

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
        label = [1 if item[0] == 1 and item[1] == 1 and item[2] == 1 else 0 for item in label]

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

# 入力データをt-SNEで可視化
def visualize_tsne(data, labels, title, output_file=None):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(data)
    # テンソルをNumPy配列に変換
    labels = labels.numpy()
    # ラベルごとの色のリストを作成
    unique_labels = np.unique(labels)
    colors = ['red', 'green', 'blue']  # 各ラベルに対する色を指定
    plt.figure(figsize=(10, 8))
    # 散布図を作成し、ラベルごとに色を設定
    for label, color in zip(unique_labels, colors):
        indices = np.where(labels == label)
        plt.scatter(tsne_data[indices, 0], tsne_data[indices, 1], c=color, label=str(label),cmap='viridis')    

    plt.title(title)

    # 凡例の表示
    plt.legend()
    plt.colorbar()
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

# モデルの推論結果と入力データを可視化
def visualize_embedding(true_gauss_tensor, true_wind_tensor, genuine_output, labels, output_file1, output_file2):
    # 入力データをnumpy配列に変換
    gauss_data = true_gauss_tensor.detach().cpu().numpy()
    wind_data = true_wind_tensor.detach().cpu().numpy()

    # 入力データとgenuine_outputを結合
    input_data = np.concatenate((gauss_data, wind_data), axis=1)
    genuine_output_data = genuine_output.detach().cpu().numpy()

    # 入力データのt-SNE可視化
    visualize_tsne(input_data, labels, 'Input Data', output_file1)

    # genuine_outputのt-SNE可視化
    visualize_tsne(genuine_output_data, labels, 'genuine_output' , output_file2)

preprocess = preprocess_for_Siamese_Net()

event_list = ["air","clap","cleaner","door","drop","fall","jump","pc","pot","sit","speak","typing","walk","window"]
train_event_dfs = []
val_event_dfs = []
test_event_dfs = []

for event in event_list:
    event_dfs = process_files(event)
    column_names = ["V(m/s)", "φ(mG)"]
    extracted_dfs = [(event,vib_df,climo_gauss_df[column_names]) for vib_df,climo_gauss_df in event_dfs]
    train_event_dfs.extend(extracted_dfs[2:])
    val_event_dfs.append(extracted_dfs[1])
    test_event_dfs.append(extracted_dfs[0])

train_wind = np.concatenate(df["V(m/s)"] for df in train_event_dfs[:][2])
train_gauss = np.concatenate(df["φ(mG)"] for df in train_event_dfs[:][2])
train_wind = np.array(train_wind)
train_gauss = np.array(train_gauss)
"""
train_wind = np.concatenate([df["V(m/s)"] for df in train_walk_dfs + train_air_dfs + train_no_dfs])
train_gauss = np.concatenate([df["φ(mG)"] for df in train_walk_dfs + train_air_dfs + train_no_dfs])
train_wind = np.array(train_wind)
train_gauss = np.array(train_gauss)
"""

# 正規化
scaler_wind = StandardScaler()
scaler_gauss = StandardScaler()
scaler_wind.fit(train_wind.reshape(-1, 1))
scaler_gauss.fit(train_gauss.reshape(-1, 1))
normalized_train_event_dfs = [(event, pd.DataFrame({
    "V(m/s)": scaler_wind.transform(df["V(m/s)"].values.reshape(-1, 1)).flatten(),
    "φ(mG)": scaler_gauss.transform(df["φ(mG)"].values.reshape(-1, 1)).flatten()
}, index=df.index)) for event,_,df in train_event_dfs]

normalized_val_event_dfs = [pd.DataFrame({
    "V(m/s)": scaler_wind.transform(df["V(m/s)"].values.reshape(-1, 1)).flatten(),
    "φ(mG)": scaler_gauss.transform(df["φ(mG)"].values.reshape(-1, 1)).flatten()
}, index=df.index) for df[2] in val_event_dfs]

normalized_test_event_dfs = [pd.DataFrame({
    "V(m/s)": scaler_wind.transform(df["V(m/s)"].values.reshape(-1, 1)).flatten(),
    "φ(mG)": scaler_gauss.transform(df["φ(mG)"].values.reshape(-1, 1)).flatten()
}, index=df.index) for df[2] in test_event_dfs]

# リスト内の各データフレームに正規化を適用
# train_walk_dfsにscalerを適用

walk_wind_list = []
walk_gauss_list = []
for df in normalized_train_walk_dfs:
    sep_walk_wind,sep_walk_gauss = generate_siamese_data(df,4,60)
    walk_wind_list.extend(sep_walk_wind)
    walk_gauss_list.extend(sep_walk_gauss)
air_wind_list = []
air_gauss_list = []
for df in normalized_train_air_dfs:
    sep_air_wind,sep_air_gauss = generate_siamese_data(df,4,60)
    air_wind_list.extend(sep_air_wind)
    air_gauss_list.extend(sep_air_gauss)
no_wind_list = []
no_gauss_list = []
for df in normalized_train_no_dfs:
    sep_no_wind,sep_no_gauss = generate_siamese_data(df,4,60)
    no_wind_list.extend(sep_no_wind)
    no_gauss_list.extend(sep_no_gauss)

val_walk_wind_list = []
val_walk_gauss_list = []
for df in normalized_val_walk_dfs:
    sep_walk_wind,sep_walk_gauss = generate_siamese_data(df,4,60)
    val_walk_wind_list.extend(sep_walk_wind)
    val_walk_gauss_list.extend(sep_walk_gauss)
val_air_wind_list = []
val_air_gauss_list = []
for df in normalized_val_air_dfs:
    sep_air_wind,sep_air_gauss = generate_siamese_data(df,4,60)
    val_air_wind_list.extend(sep_air_wind)
    val_air_gauss_list.extend(sep_air_gauss)
val_no_wind_list = []
val_no_gauss_list = []
for df in normalized_val_no_dfs:
    sep_no_wind,sep_no_gauss = generate_siamese_data(df,4,60)
    val_no_wind_list.extend(sep_no_wind)
    val_no_gauss_list.extend(sep_no_gauss)

test_walk_wind_list = []
test_walk_gauss_list = []
for df in normalized_test_walk_dfs:
    sep_walk_wind,sep_walk_gauss = generate_siamese_data(df,4,60)
    test_walk_wind_list.extend(sep_walk_wind)
    test_walk_gauss_list.extend(sep_walk_gauss)
test_air_wind_list = []
test_air_gauss_list = []
for df in normalized_test_air_dfs:
    sep_air_wind,sep_air_gauss = generate_siamese_data(df,4,60)
    test_air_wind_list.extend(sep_air_wind)
    test_air_gauss_list.extend(sep_air_gauss)
test_no_wind_list = []
test_no_gauss_list = []
for df in normalized_test_no_dfs:
    sep_no_wind,sep_no_gauss = generate_siamese_data(df,4,60)
    test_no_wind_list.extend(sep_no_wind)
    test_no_gauss_list.extend(sep_no_gauss)

#ランダムに200個,200個,1000個選択する
walk_wind_list = random.sample(walk_wind_list, 100)
walk_gauss_list = random.sample(walk_gauss_list, 100)
air_wind_list = random.sample(air_wind_list, 100)
air_gauss_list = random.sample(air_gauss_list, 100)
no_wind_list = random.sample(no_wind_list, 500)
no_gauss_list = random.sample(no_gauss_list, 500)

#ランダムに80個,80個,400個選択する
val_walk_wind_list = random.sample(val_walk_wind_list, 40)
val_walk_gauss_list = random.sample(val_walk_gauss_list, 40)
val_air_wind_list = random.sample(val_air_wind_list, 40)
val_air_gauss_list = random.sample(val_air_gauss_list, 40)
val_no_wind_list = random.sample(val_no_wind_list, 200)
val_no_gauss_list = random.sample(val_no_gauss_list, 200)

#ランダムに80個,80個,400個選択する
test_walk_wind_list = random.sample(test_walk_wind_list, 40)
test_walk_gauss_list = random.sample(test_walk_gauss_list, 40)
test_air_wind_list = random.sample(test_air_wind_list, 40)
test_air_gauss_list = random.sample(test_air_gauss_list, 40)
test_no_wind_list = random.sample(test_no_wind_list, 200)
test_no_gauss_list = random.sample(test_no_gauss_list, 200)

wind_a_set,wind_b_set,gauss_a_set,gauss_b_set,labels = generate_npy_from_siamese_data(walk_wind_list,
                                                                                        walk_gauss_list,
                                                                                        no_wind_list,
                                                                                        no_gauss_list,
                                                                                        air_wind_list,
                                                                                        air_gauss_list)
val_wind_a_set,val_wind_b_set,val_gauss_a_set,val_gauss_b_set,val_labels = generate_npy_from_siamese_data(val_walk_wind_list,
                                                                                        val_walk_gauss_list,
                                                                                        val_no_wind_list,
                                                                                        val_no_gauss_list,
                                                                                        val_air_wind_list,
                                                                                        val_air_gauss_list)
test_wind_a_set,test_wind_b_set,test_gauss_a_set,test_gauss_b_set,test_labels = generate_npy_from_siamese_data(test_walk_wind_list,
                                                                                        test_walk_gauss_list,
                                                                                        test_no_wind_list,
                                                                                        test_no_gauss_list,
                                                                                        test_air_wind_list,
                                                                                        test_air_gauss_list)
#npyファイルに変換
datadir = "data/train-npy/"

np.save(datadir + 'train_wind_set', wind_a_set)
np.save(datadir + 'train_gauss_set', gauss_a_set)
np.save(datadir + 'train_labels', labels)
np.save(datadir + 'val_wind_set', val_wind_a_set)
np.save(datadir + 'val_gauss_set', val_gauss_a_set)
np.save(datadir + 'val_labels', val_labels)
np.save(datadir + 'test_wind_set', test_wind_a_set)
np.save(datadir + 'test_gauss_set', test_gauss_a_set)
np.save(datadir + 'test_labels', test_labels)

datadir = "data/train-npy/"
checkpoints_dir = "data/checkpoints/"
logs_dir = "data/logs/"

train_gauss = np.load(datadir + 'train_gauss_set.npy')
train_wind = np.load(datadir + 'train_wind_set.npy')
train_label = np.load(datadir + 'train_labels.npy')
val_gauss = np.load(datadir + 'val_gauss_set.npy')
val_wind = np.load(datadir + 'val_wind_set.npy')
val_label = np.load(datadir + 'val_labels.npy')
test_gauss = np.load(datadir + 'test_gauss_set.npy')
test_wind = np.load(datadir + 'test_wind_set.npy')
test_label = np.load(datadir + 'test_labels.npy')

n_max_gpus = torch.cuda.device_count()
print(f'{n_max_gpus} GPUs available')
n_gpus = min(2, n_max_gpus)
print(f'Using {n_gpus} GPUs')


#識別学習に用いるone-hot表現のラベルを作成
train_one_hot_labels = torch.zeros(len(train_gauss), 3, dtype=torch.float)
val_one_hot_labels = torch.zeros(len(val_gauss), 3, dtype=torch.float)
test_one_hot_labels = torch.zeros(len(test_gauss), 3, dtype=torch.float)
for step, genuine_label in enumerate(train_label[:,0]):
    if genuine_label == 0:
        train_one_hot_labels[step]=torch.tensor([1,0,0],dtype=torch.float)
    if genuine_label == 1:
        train_one_hot_labels[step]=torch.tensor([0,1,0],dtype=torch.float)
    if genuine_label == 2:
        train_one_hot_labels[step]=torch.tensor([0,0,1],dtype=torch.float)
for step, genuine_label in enumerate(val_label[:,0]):
    if genuine_label == 0:
        val_one_hot_labels[step]=torch.tensor([1,0,0],dtype=torch.float)
    if genuine_label == 1:
        val_one_hot_labels[step]=torch.tensor([0,1,0],dtype=torch.float)
    if genuine_label == 2:
        val_one_hot_labels[step]=torch.tensor([0,0,1],dtype=torch.float)
for step, genuine_label in enumerate(test_label[:,0]):
    if genuine_label == 0:
        test_one_hot_labels[step]=torch.tensor([1,0,0],dtype=torch.float)
    if genuine_label == 1:
        test_one_hot_labels[step]=torch.tensor([0,1,0],dtype=torch.float)
    if genuine_label == 2:
        test_one_hot_labels[step]=torch.tensor([0,0,1],dtype=torch.float)

traindataset = DummyDataset(train_gauss,train_wind,train_one_hot_labels)
valdataset = DummyDataset(val_gauss,val_wind,val_one_hot_labels)


epochs = 10
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
file1 = open(logs_dir + "train_FC_accuracies.txt","w")
file2 = open(logs_dir + 'val_FC_accuracies.txt','w')
"""
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
        #_, y_targets = train_labels.clone().max(dim=1)
        #output_path1 = "plot/data-tSNE.png"
        #output_path2 = "plot/vector-tSNE.png"
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

model_path = checkpoints_dir + "dist_model_" + str(epoch) + ".pt"
model.load_state_dict(torch.load(model_path))
"""
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class FC(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64,num_classes)
        self.dropout1 = nn.Dropout(0.3)
        self.bn_block_1 = nn.BatchNorm1d(64)
        self.relu = nn.PReLU()

    def forward(self, gauss_input,wind_input):
        C_x = self.encoder(gauss_input,wind_input)
        x = self.fc1(C_x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.bn_block_1(x)
        x = self.fc2(x)
        return x,C_x

# Create an instance of the CNN model
identify_model = FC(num_classes=3)
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
        train_outputs,_ = identify_model(train_gauss_tensor.to(device1),train_wind_tensor.to(device1))
        #calculate cross entropy loss
        _, train_y_targets = train_labels.clone().max(dim=1)
        train_cross_en_loss = criterion(train_outputs.requires_grad_(True), train_y_targets.long().to(device1))
        train_cross_en_losses.append(train_cross_en_loss.cpu().detach().numpy())
        #calculate accuracy
        train_outputs_softmax = torch.softmax(train_outputs,dim=1)
        train_predicted_classes = torch.argmax(train_outputs_softmax, dim=1)
        # one-hot表現に変換
        train_predicted_labels = torch.zeros(train_outputs.size(0), 3)
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
            val_outputs,embedding = identify_model(val_gauss_tensor.to(device1), val_wind_tensor.to(device1))
            #calculate loss
            _, y_val_targets = val_labels.clone().max(dim=1)
            #output_path1 = "plot/data-tSNE.png"
            #output_path2 = "plot/vector-tSNE.png"
            #visualize_embedding(val_gauss_tensor, val_wind_tensor, embedding, y_val_targets, output_path1, output_path2)
            #sys.exit()
            val_cross_en_loss = criterion(val_outputs.requires_grad_(True), y_val_targets.long().to(device1))
            val_cross_en_losses.append(val_cross_en_loss.cpu().detach().numpy())
            #calculate accuracy
            val_outputs_softmax = torch.softmax(val_outputs,dim=1)
            val_predicted_classes = torch.argmax(val_outputs_softmax, dim=1)
            # one-hot表現に変換
            val_predicted_labels = torch.zeros(val_outputs.size(0), 3)
            val_predicted_labels.scatter_(1, val_predicted_classes.cpu().unsqueeze(1), 1)
            val_correct = (val_predicted_labels.to(device1) == val_labels.to(device1)).sum().item()
            val_total = val_labels.numel()
            val_accuracy = val_correct / val_total
            val_class_accuracies.append(val_accuracy)

        print(f"EPOCH {epoch}| Val: class loss  {np.mean(val_cross_en_losses)} | val accuracy {np.mean(val_class_accuracies)} ")
        file2.write("%s, %s, %s, %s, %s, %s\n" % (str(epoch), "val_cross_en_loss", str(np.mean(val_cross_en_losses)), "val_accuracy", str(np.mean(val_class_accuracies)), now_time))
file1.close()
file2.close()

testdataset = DummyDataset(test_gauss,test_wind,test_one_hot_labels)
batch_size = 1000
test_dataloader = DataLoader(testdataset, batch_size = batch_size, shuffle=True)
#モデル構築
class_model_path = checkpoints_dir + "class_model_" + str(class_epochs) + ".pt"
class_checkpoint = torch.load(class_model_path)
#model.load_state_dict(torch.load(model_path))
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
        test_outputs,_ = identify_model(test_gauss_tensor.to(device1), test_wind_tensor.to(device1))
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
file3 = open(logs_dir + "test_FC_metrics.txt","w")
file3.write("%s,%s,\n%s,%s,\n%s,%s,\n%s,%s,\n" %("Accuracy",str(acc),"Precision",str(prec),"Recall",str(recall),"F1 score",str(f1)))
file3.close()
