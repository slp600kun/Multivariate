#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:10:08 2021

@author: kawaguchi
"""
import scipy.io as sc
import numpy as np
import os

def convert_mat_to_list(mat_dir:str,feature_type:str) -> None:
    """
    複数のmatファイルをlistに変換する関数
    args: 
        - mat_dir(str): matファイルが格納されているディレクトリ名
    returns:
        - None

    """

    mat_files = os.listdir(mat_dir)
    mat2_a=[]
    mat2_b=[]
    mat2_y=[]
    for i in range(len(mat_files)):
            a_path = mat_files[i]
            temp1 = sc.loadmat(mat_dir + a_path)
            feat1 = np.array(temp1[feature_type]).reshape(999,50)
            for j in range(len(mat_files)):
                b_path = mat_files[j]
                temp2 = sc.loadmat(mat_dir + b_path)
                feat2 = np.array(temp2[feature_type]).reshape(999,50)
                mat2_a.append(feat1)
                mat2_b.append(feat2)
                if(a_path[0]==b_path[0]):
                    mat2_y.append(ord(a_path[0])-48)
                else:
                    mat2_y.append(-1)

    return mat2_a,mat2_b,mat2_y

def data_generate(mat1_dir,mat1_feat,mat2_dir,mat2_feat):
    """
    
    """


    mat1_a,mat1_b,mat1_y = convert_mat_to_list(mat1_dir,mat1_feat)
    mat2_a,mat2_b,mat2_y = convert_mat_to_list(mat2_dir,mat2_feat)
    label=[]

    for i in range(len(mat2_a)):
        if(mat1_y[i] == -1 or mat2_y[i] == -1):
            label.append(0)
        elif(mat1_y[i] == mat2_y[i]):
            label.append(1)
        else:
            label.append(0)

    return mat1_a,mat2_a,mat1_b,mat2_b,label