#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:10:08 2021

@author: shakeel
"""
import numpy as np
import features
import audiodata
import os

geo_dir = "/misc/export3/shakeel/multimodal/algo2/train-data/data-set-2/geos/"
audio_dir = "/misc/export3/shakeel/multimodal/algo2/train-data/data-set-2/audios/"
datadir = "train-data/"

#audiofiles = os.listdir(audio_dir)
geo_a,aud_a,geo_b,aud_b,labels = features.data_generate(geo_dir,audio_dir)

#aud_a,aud_b,aud_y=audiodata.data(audio_dir)

np.save(datadir + 'geophone_a_set_2', geo_a)
np.save(datadir + 'geophone_b_set_2', geo_b)
np.save(datadir + 'audio_a_set_2', aud_a)
np.save(datadir + 'audio_b_set_2', aud_b)
np.save(datadir + 'labels_set_2', labels)
#a = np.load(datadir + 'geophone_a.npy')
#b = np.load(datadir + 'audio_a.npy')
