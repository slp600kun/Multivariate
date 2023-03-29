#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test-train-shakeel/format_data.py
#
# format time series data for training
#

import os
import re
import pandas as pd
import datetime as dt
import pandas as pd
import numpy as np
import scipy
import matlab.engine

from sklearn.preprocessing import minmax_scale

#linux内のmatlabを起動する
#mat = matlab.engine.start_matlab()
#mat.cd(r'matlab', nargout=0)


class preprocess_for_Siamese_Net():
    """
    siamese network用のデータ前処理クラス
    """


    def __init__(self):
        super()

    @classmethod
    def convert_csv_to_mergedcsv(self, climo_csv_path:str, gauss_csv_path:str) -> pd:
        """
        2つのセンサーの時系列csvデータを同時刻部のみを切り出しmergeする

        arg:
            - climo_csv_path:climomasterデータのcsvパス
            - gauss_csv_path:milli gauss meterデータのcsvパス
        return:
            - merged_df:二つのセンサーの時系列csvデータを同時刻部のみを切り出しmergeしたデータフレーム
        """

        def climo_momenttime(x_linespace:int,beginning_time:str,sampling_interval:int)-> dt:
            """
            climomaster.dfの各データ行の日付型を生成

            args:
                - x_linespace: 等間隔の数列
                - beginning_time: 開始時刻
                - sampleing_interval: サンプリング間隔

            return:
                - moment_time: 日付型
            """
            
            #開始時刻の文字列からデータの日付型を取得
            beginning_time = beginning_time.replace('#','')
            beginning_time_list = re.split('[/: ]',beginning_time)
            beginning_year = int(beginning_time_list[2])
            beginning_month = int(beginning_time_list[1])
            beginning_day = int(beginning_time_list[0])
            beginning_hour = int(beginning_time_list[3])
            beginning_minite = int(beginning_time_list[4])
            beginning_second = int(beginning_time_list[5])
            beginning_time = dt.datetime(beginning_year,beginning_month,beginning_day,beginning_hour,beginning_minite,beginning_second)

            #サンプリング間隔
            elapsed_time = x_linespace * sampling_interval
            moment_time = beginning_time + dt.timedelta(seconds = elapsed_time)
            return moment_time


        def gauss_momenttime(time:str)-> dt:
            """
            milli gauss meter.dfの各データ行の日付型を生成

            arg:
                -time:時刻

            return:
                - moment_time: 日付型
            """
            moment_time_list = re.split('[/: ]',time)
            year = int(moment_time_list[0])
            month = int(moment_time_list[1])
            day = int(moment_time_list[2])
            hour = int(moment_time_list[3])
            minute = int(moment_time_list[4])
            second = int(moment_time_list[5])
            moment_time = dt.datetime(year,month,day,hour,minute,second)
            return moment_time
        
        #パスからデータフレームを取得
        climo_csv_df = pd.read_csv(climo_csv_path,names = ["No.","V(m/s)","T(C)","H(%RH)","A(hPa)","Atom.F"])

        #climomaster.csvの先頭部のinfo情報のみをclimo_info_dictに抽出
        climo_info_df = climo_csv_df[:13]
        climo_info_df =  climo_info_df.rename(columns={'No.': 'info','V(m/s)':'data'})
        climo_info_df = climo_info_df.drop(["T(C)","H(%RH)","A(hPa)","Atom.F"],axis=1)
        climo_info_dict = dict(zip(climo_info_df['info'], climo_info_df['data']))

        #開始時間とサンプリング間隔を抽出
        beginning_time = climo_info_dict['[Measurement beginning time]']
        sampling_interval = int(climo_info_dict['[S-TIME(S) -- Sampling interval(s)]'])

        #climomaster.csvのデータ部の情報を加工し、climo_dfに抽出
        climo_df = climo_csv_df[19:]
        climo_df =  climo_df.rename(columns={'No.': 'Time'})
        climo_df = climo_df.drop(["A(hPa)","Atom.F"],axis=1)
        climo_df = climo_df.astype({'Time':int, 'V(m/s)':float, "T(C)":float ,"H(%RH)":float})
        
        #一行ずつ時間を更新
        for i in range(len(climo_df)):
            number = sampling_interval * (i+1)
            climo_df.loc[i+19,"Time"] = climo_momenttime(number,beginning_time,sampling_interval)
        #行の部分に時間をセットする
        climo_df = climo_df.set_index('Time')

        #milli gauss meterのデータフレームを取得
        gauss_df = pd.read_csv(gauss_csv_path, encoding = 'shift-jis')

        #gauss.csvのデータ部の情報を加工し、gauss_dfを取得
        gauss_df['Time'] = gauss_df['年月日'].str.cat(gauss_df['時刻'], sep = ' ')
        gauss_df = gauss_df.drop(columns = {'年月日','時刻'})
        gauss_df = gauss_df.rename(columns = {'データ01(mG)':'φ(mG)'})
        #全ての"Time"行にgauss_momenttime関数を適用
        gauss_df['Time'] = gauss_df['Time'].map(gauss_momenttime)
        #行の部分に時間をセットする
        gauss_df = gauss_df.set_index('Time')

        #両方のcsvをmergeし、時刻の共通部のみを切り出す
        merged_df = pd.merge(climo_df,gauss_df,left_index=True,right_index = True)
        merged_df = merged_df.dropna()
        merged_df.plot(subplots = True)
        
        return merged_df
    
    @classmethod
    def normalization(data_array:np)->np:
        """
        時系列データ(numpy配列)を配列内で最大値最小値を取り、正規化する

        args:
            - data_array(np): データのnumpy配列
        return:
            - rms_data(np):正規化処理を加えた後のデータのnumpy配列
        """
        normalizated_data = minmax_scale(data_array)
        return normalizated_data

    @classmethod
    def rms(data_array:np)->np:
        """
        時系列データ(numpy配列)に実効値処理を加える

        args:
            - data_array(np): データのnumpy配列
        return:
            - rms_data(np):実効値処理を加えた後のデータのnumpy配列
        """
        square_data_array = data_array**2
        abs_square_data_array = np.abs(square_data_array)
        sum_abs_square_data = np.sum(abs_square_data_array)
        sum_mean_square_data = sum_abs_square_data / len(data_array)
        rms_data = np.sqrt(data_array / sum_mean_square_data)
        return rms_data
    
    @classmethod
    def convert_mergedcsv_to_matfile(self,csv_path:str,
                                        is_wind_vel_converted: bool = True,
                                        is_temp_converted: bool = False,
                                        is_humid_converted: bool = False,
                                        is_gauss_converted: bool = True):    
        """
        メルスペクトログラム変換する関数
        """

        #データの取り込み
        merged_data_df = pd.read_csv(csv_path,names = ["Time","V(m/s)","T(C)","H(%RH)","φ(mG)"],skiprows=1)
        
        #風速をmatファイルに変換
        if is_wind_vel_converted:
            wind_vel_df = merged_data_df['V(m/s)']
            wind_vel_array = np.array(wind_vel_df.values[1:], dtype = 'float')
            mel_spect_wind_vel_data = mat.mel_spectrogram_bad(matlab.double(wind_vel_array),matlab.double(2900))  
            scipy.io.savemat('data/mat/wind_vel/wind'+ csv_path.replace('.csv','') + '.mat', {'wind_feat':mel_spect_wind_vel_data})  
            print("convert wind velocity data to mat file.") 

        #温度をmatファイルに変換
        if is_temp_converted:
            temp_df = merged_data_df['T(C)']
            temp_array = np.array(temp_df.values[1:], dtype = 'float')
            mel_spect_temp_data = mat.mel_spectrogram_bad(matlab.double(temp_array),matlab.double(2900))  
            scipy.io.savemat('data/mat/temp/temp'+ csv_path.replace('.csv','') + '.mat', {'temp_feat':mel_spect_temp_data})   
            print("convert temperature data to mat file.")  

        #湿度をmatファイルに変換
        if is_humid_converted:
            humid_df = merged_data_df['H(%RH)']
            humid_array = np.array(humid_df.values[1:], dtype = 'float')
            mel_spect_humid_data = mat.mel_spectrogram_bad(matlab.double(humid_array),matlab.double(2900))  
            scipy.io.savemat('data/mat/humid/humid'+ csv_path.replace('.csv','') + '.mat', {'humid_feat':mel_spect_humid_data}) 
            print("convert humidity data to mat file.")   
        
        #磁束密度をmatファイルに変換
        if is_gauss_converted:
            gauss_df = merged_data_df['H(%RH)']
            gauss_array = np.array(gauss_df.values[1:], dtype = 'float')
            mel_spect_gauss_data = mat.mel_spectrogram_bad(matlab.double(gauss_array),matlab.double(2900))  
            scipy.io.savemat('data/mat/gauss/gauss'+ csv_path.replace('.csv','') + '.mat', {'gauss_feat':mel_spect_gauss_data})  
            print("convert milli Gauss data to mat file.")  

    def generate_npy_for_siamese(mat1_dir,mat1_feat,mat2_dir,mat2_feat):
        """
        siamese networks用に2対のデータセットとラベルのnpyファイルを生成
        args:
            - mat1_dir(str): matファイルが格納されているディレクトリ
            - mat1_feat: ['wind_feat','temp_feat','humid_feat','gauss_feat']のどれか、mat1_dirの種類による
            - mat2_dir(str): matファイルが格納されているディレクトリ(もう一組)
            - mat2_feat: ['wind_feat','temp_feat','humid_feat','gauss_feat']のどれか、mat1_dirの種類による
        """

        def convert_mat_to_list(mat_dir:str,feature_type:str):
            """
            ディレクトリ内のmatファイルの組み合わせを全てlistに変換する関数
            args: 
                - mat_dir(str): matファイルが格納されているディレクトリ名
                - feature_type(str): ['wind_feat','temp_feat','humid_feat','gauss_feat']のどれか、mat_dirの種類による
            """

            mat_files = os.listdir(mat_dir)
            mat2_a = []
            mat2_b = []
            mat2_y = []
            for i in range(len(mat_files)):
                    a_path = mat_files[i]
                    temp1 = scipy.io.loadmat(mat_dir + a_path)
                    feat1 = np.array(temp1[feature_type]).reshape(999,50)
                    for j in range(len(mat_files)):
                        b_path = mat_files[j]
                        temp2 = scipy.io.loadmat(mat_dir + b_path)
                        feat2 = np.array(temp2[feature_type]).reshape(999,50)
                        mat2_a.append(feat1)
                        mat2_b.append(feat2)
                        if(a_path[0]==b_path[0]):
                            mat2_y.append(ord(a_path[0])-48)
                        else:
                            mat2_y.append(-1)

            return mat2_a,mat2_b,mat2_y

        mat1_a,mat1_b,mat1_y = convert_mat_to_list(mat1_dir,mat1_feat)
        mat2_a,mat2_b,mat2_y = convert_mat_to_list(mat2_dir,mat2_feat)
        label=[]

        #ラベリング
        for i in range(len(mat2_a)):
            if(mat1_y[i] == -1 or mat2_y[i] == -1):
                label.append(0)
            elif(mat1_y[i] == mat2_y[i]):
                label.append(1)
            else:
                label.append(0)

        #npyファイルに変換s
        datadir = "data/train-npy/"
        np.save(datadir + mat1_feat + '_a_set', mat1_a)
        np.save(datadir + mat1_feat + '_b_set', mat1_b)
        np.save(datadir + mat2_feat + '_a_set', mat2_a)
        np.save(datadir + mat2_feat + '_b_set', mat2_b)
        np.save(datadir + 'labels', label)
        return

