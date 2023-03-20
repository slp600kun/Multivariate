#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:10:08 2021

@author: shakeel
"""
import audiodata
import geophonedata

def data_generate(geo_dir,audio_dir):
    geo_a,geo_b,geo_y=geophonedata.data(geo_dir)
    aud_a,aud_b,aud_y=audiodata.data(audio_dir)
    label=[]
    for i in range(len(aud_a)):
        if(geo_y[i]==-1 or aud_y[i] == -1):
            label.append(0)
        elif(geo_y[i]==aud_y[i]):
            label.append(1)
        else:
            label.append(0)
    return geo_a,aud_a,geo_b,aud_b,label
