#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:10:08 2021

@author: shakeel
"""
import scipy.io as sc
import numpy as np
import os

def data(geo_dir):
    geophone_files = os.listdir(geo_dir)
    geo_a=[]
    geo_b=[]
    geo_y=[]
    for i in range(len(geophone_files)):
            ga_path=geophone_files[i]
            temp1 = sc.loadmat(geo_dir + ga_path)
            feat1 = np.array(temp1['feat']).reshape(999,50)
            for j in range(len(geophone_files)):
                gb_path=geophone_files[j]
                temp2 = sc.loadmat(geo_dir + gb_path)
                feat2 = np.array(temp2['feat']).reshape(999,50)
                geo_a.append(feat1)
                geo_b.append(feat2)
                if(ga_path[0]==gb_path[0]):
                    geo_y.append(ord(ga_path[0])-48)
                else:
                    geo_y.append(-1)
    return geo_a,geo_b,geo_y
