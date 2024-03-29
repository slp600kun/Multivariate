#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:10:08 2021

@author: shakeel
"""
import scipy.io as sc
import numpy as np
import os

def data(audio_dir):
    audio_files = os.listdir(audio_dir)
    aud_a=[]
    aud_b=[]
    aud_y=[]
    for i in range(len(audio_files)):
            a_path=audio_files[i]
            temp1 = sc.loadmat(audio_dir + a_path)
            feat1 = np.array(temp1['feat']).reshape(999,50)
            for j in range(len(audio_files)):
                b_path=audio_files[j]
                temp2 = sc.loadmat(audio_dir + b_path)
                feat2 = np.array(temp2['feat']).reshape(999,50)
                aud_a.append(feat1)
                aud_b.append(feat2)
                if(a_path[0]==b_path[0]):
                    aud_y.append(ord(a_path[0])-48)
                else:
                    aud_y.append(-1)
    return aud_a,aud_b,aud_y