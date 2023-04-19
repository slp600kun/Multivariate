import sys
import os
import re
import numpy as np
from preprocess_data import preprocess_for_Siamese_Net
from train import ConvLayer2D,windEncoder,CombinedEncoder,DummyDataset,ContrastiveLoss,SVM_for_two_dim
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

    feat1_a,feat1_b,feat1_y = labeling_for_action(action_feat1,not_action_feat1)
    feat2_a,feat2_b,feat2_y = labeling_for_action(action_feat2,not_action_feat2)

    label=[]

    #ラベリング
    for i in range(len(feat2_a)):
        if(feat1_y[i] == -1 or feat2_y[i] == -1):
            label.append(0)
        #損失関数計算前に、-1を1に直す(svmにかける直前)
        elif(feat1_y[i] == feat2_y[i]):
            if feat1_y[i] == 1:
                label.append(1)
            elif feat1_y[i] == 0:
                label.append(-1)
        else:
            label.append(0)

    # Combine the arrays into a list of tuples
    combined = list(zip(feat1_a, feat1_b, feat2_a, feat2_b, label))

    # Shuffle the list using random.shuffle()
    random.shuffle(combined)

    # Unpack the shuffled tuples back into separate arrays
    feat1_a_set, feat1_b_set, feat2_a_set, feat2_b_set, labels = zip(*combined)

    return feat1_a_set, feat1_b_set, feat2_a_set, feat2_b_set, labels

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

    walk_wind_vel,walk_gauss,no_wind_vel,no_gauss = generate_siamese_data(walk_merged_df,no_merged_df,4,30,300*(i+1))

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

datadir = "data/train-npy/"
checkpoints_dir = "data/checkpoints/"
logs_dir = "data/logs/"

true_gauss = np.load(datadir + 'gauss_a_set.npy')
true_wind = np.load(datadir + 'wind_a_set.npy')
wrong_gauss = np.load(datadir + 'gauss_b_set.npy')
wrong_wind = np.load(datadir + 'wind_b_set.npy')
label = np.load(datadir + 'labels.npy')

traindataset = DummyDataset(true_gauss[0:750000],true_wind[0:750000],wrong_gauss[0:750000],
                       wrong_wind[0:750000],label[0:750000])

valdataset = DummyDataset(true_gauss[750000:1000000],true_wind[750000:1000000],wrong_gauss[750000:1000000],
                       wrong_wind[750000:1000000],label[750000:1000000])

epochs = 3
batch_size = 1000
train_dataloader = DataLoader(traindataset, batch_size = batch_size, shuffle=True)
val_dataloader = DataLoader(valdataset, batch_size = batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#loss_fn = nn.CosineEmbeddingLoss().to(device)
loss_fn = ContrastiveLoss().to(device)
model = CombinedEncoder().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

#SVMマシンのモデル定義
svm_model = SVM_for_two_dim().to(device)
svm_loss_fn = torch.nn.HingeEmbeddingLoss().to(device)
svm_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train() 


torch.set_grad_enabled(True)
print("STARING TO TRAIN MODEL")
file1 = open(logs_dir + "training_accuracies.txt","w")
file2 = open(logs_dir + 'validation_accuracies.txt','w')
file3 = open(logs_dir + 'svm_accuracies.txt' ,'w')
for epoch in range(1, epochs+1):

    model.train()
    #svm学習に必要な配列
    feature_vector_train = []
    feature_label_train = []
    feature_vector_val = []
    feature_label_val = []
    
    steps_losses = []
    steps_accu = []

    model_checkpoints = checkpoints_dir + "model_" + str(epoch) + ".pt"
    for steps, (true_gauss_tensor, true_wind_tensor, wrong_gauss_tensor, wrong_wind_tensor, labels) in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
        optimizer.zero_grad() 
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

        #2次元特徴量ベクトルを別の配列に格納
        genuine_np = genuine_output[1].cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        for i in range(len(genuine_np)):
            if labels_np[i] == 1:
                #特徴量ベクトルとラベルを配列に追加
                feature_vector_train.append(torch.tensor(genuine_np[i]))
                feature_label_train.append(1)
            if labels_np[i] == -1:
                #特徴量ベクトルとラベルを配列に追加
                feature_vector_train.append(torch.tensor(genuine_np[i]))
                feature_label_train.append(-1)

        #-1→1に変換(距離学習を行うため) 
        abs_labels = torch.abs(labels).int()

        loss,y_pred = loss_fn(genuine_output[0], forged_output[0], abs_labels.to(device))
        steps_losses.append(loss.cpu().detach().numpy())
        prediction = (y_pred.cpu().detach().numpy()>0.4).astype(int)
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
            true_gauss_tensor = torch.unsqueeze(true_gauss_tensor, dim = 3)
            true_wind_tensor = torch.unsqueeze(true_wind_tensor, dim = 3)
            wrong_gauss_tensor = torch.unsqueeze(wrong_gauss_tensor, dim = 3)
            wrong_wind_tensor = torch.unsqueeze(wrong_wind_tensor, dim = 3)

            genuine_output = model(true_gauss_tensor.to(device), true_wind_tensor.to(device))
            forged_output = model(wrong_gauss_tensor.to(device), wrong_wind_tensor.to(device))

            #特徴量ベクトルを別の配列に格納
            genuine_np = genuine_output[1].cpu().detach().numpy()
            labels_np = labels.cpu().detach().numpy()
            for i in range(len(genuine_np)):
                if labels_np[i] == 1:
                    #特徴量ベクトルとラベルを配列に追加
                    feature_vector_val.append(torch.tensor(genuine_np[i]))
                    feature_label_val.append(1)
                if labels_np[i] == -1:
                    #特徴量ベクトルとラベルを配列に追加
                    feature_vector_val.append(torch.tensor(genuine_np[i]))
                    feature_label_val.append(-1)
            
            #-1→1に変換(距離学習を行うため)
            abs_labels = torch.abs(labels).int()

            loss,y_pred = loss_fn(genuine_output[0], forged_output[0], labels.to(device))
            prediction = (y_pred.cpu().detach().numpy()>0.4).astype(int)
            accuracy = accuracy_score(labels,prediction)
            steps_accu.append(accuracy)
            steps_losses.append(loss.cpu().numpy())
        print(f"EPOCH {epoch}| Validation:  loss {np.mean(steps_losses)}| accuracy {np.mean(steps_accu)} {now_time}")
        file2.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "val_loss", str(np.mean(steps_losses)), "val_accuracy", str(np.mean(steps_accu)), str(now_time)))
file1.close()
file2.close()

#識別学習(svm)
#テンソルに変換
vector_train = torch.stack(feature_vector_train)
vector_val = torch.stack(feature_vector_val)
label_train = torch.tensor(feature_label_train)
label_val = torch.tensor(feature_label_val)

# データセットを作成
svm_train_dataset = TensorDataset(vector_train, label_train)
svm_val_dataset = TensorDataset(vector_val, label_val)
# データローダーを作成
svm_train_loader = DataLoader(svm_train_dataset,  batch_size= 128,shuffle=True)
svm_val_loader = DataLoader(svm_val_dataset, batch_size= 128,shuffle=True)

svm_model.train()
for epoch in range(100):
    
    svm_steps_losses = []
    svm_steps_accu = []

    for steps, (inputs, labels) in enumerate(svm_train_loader):
        outputs = svm_model(inputs.to(device))
        svm_loss = svm_loss_fn(outputs, labels.to(device))
        svm_steps_losses.append(svm_loss.cpu().detach().numpy())
        svm_loss.backward()
        svm_optimizer.step()
    print(f'Epoch {epoch}, train loss: {np.mean(svm_steps_losses)}, ')

    svm_model.eval()
    with torch.no_grad():
        correct_eval = 0
        total_eval = 0
        for eval_vec, eval_labels in svm_val_loader:
            output_eval = svm_model(eval_vec.to(device))
            predicted_eval = torch.sign(output_eval).squeeze().long()
            total_eval += eval_labels.to(device).size(0)
            correct_eval += (predicted_eval == eval_labels.to(device)).sum().item()
        accuracy_eval = correct_eval / total_eval

    print(f'Epoch {epoch}, Accuracy on evaluation data: {accuracy_eval}')
    file3.write("%s , %s, %s, %s, %s, %s\n" % (str(epoch), "loss", str(np.mean(svm_steps_losses)), "val_accuracy", str(accuracy_eval), str(now_time)))
file3.close()