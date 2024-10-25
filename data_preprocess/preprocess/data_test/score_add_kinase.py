#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import sys
import glob
import pickle
import shutil
import random
import datetime
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from scipy import sparse
from tensorflow.keras.models import load_model
#from keras.models import load_model
from pandas import DataFrame
import pandas as pd


cf=400
h=50
datm=200
damino=200
dchrg=200
ddist=200
A=35
D=50
C=50
R=34
kc=6
kp=2
max_atom_num=100
zi = (datm + dchrg + ddist) * (kc +kp) + damino * kp

def get_mask_mat(mask):
    return np.array([np.vstack((np.ones((i, 1, cf)), np.zeros((max_atom_num - i, 1, cf)))) for i in mask])

def read_single_no_grid_pickle(task_folder, cmpd_lib):
    # task_folder = "/home/zdx/kinase_project/kinformation_20190724_feature_reshape/A3FQ16/codi"
    input_list = []
    if cmpd_lib==None:
        data_name = '{}/tmp/train.no_grid.pickle'.format(task_folder)
    else:
        data_name = '{}/tmp/{}.no_grid.pickle'.format(task_folder, cmpd_lib)
    with open(data_name, 'rb') as f:
        one_target_feature = pickle.load(f)
    for i in [1,2,3,4]:
        input_list.append(one_target_feature[i])

    mask =  get_mask_mat(one_target_feature[5])
    input_list.append(mask)
    name_list_tmp = one_target_feature[0]
    name_list = []
    for item in name_list_tmp:
        name_list.append(item.split("_pose_")[0])
    try:
        y = one_target_feature[6][:,1]
    except:
        y = [-1]
    return input_list, y, name_list

def predict_one_target(task_folder, cmpd_lib=None):
    # task_folder = task_folders[0]
    Zinput, y, name_list = read_single_no_grid_pickle(task_folder, cmpd_lib)
    results = model.predict(Zinput, batch_size=1024).flatten()
    # create DataFrame
    if y[0]==-1:
        result_dict = {"#ID":name_list, "score":results}
        result_df = DataFrame(result_dict)
        result_df.sort_values("score", ascending=False, inplace=True)
        result_df = result_df.reset_index()
        del result_df['index']  
    else:
        result_dict = {"#ID":name_list, "score":results, "label":y}
        result_df = DataFrame(result_dict)
        result_df.sort_values("score", ascending=False, inplace=True)
        result_df = result_df.reset_index()
        del result_df['index']   
    #print(result_df)
    return result_df

def main(path):
    global model
    failed = []
    MUV_kinase_task_folders = glob.glob(path)
    for i in range(3):
        # i = 0
        for j in range(0, 359):
            # j = 47
            model_path = '../add_kinase/%d/models/%s.h5' % (i, j)
            score_folder = '../add_kinase/%d/MUV_scores/%s' % (i, j)
            model = load_model(model_path)
            if not os.path.exists(score_folder):
                os.makedirs(score_folder)
            for task_folder in MUV_kinase_task_folders:
                # task_folder = MUV_kinase_task_folders[0]
                target_name = task_folder.split('/')[-1]
                try:
                    df = predict_one_target(task_folder)
                    score_path = '%s/%s.score' % (score_folder, target_name)
                    df.to_csv(score_path, index=False)
                except:
                    failed.append(model_path)
            K.clear_session()
    with open('../add_kinase/failed', 'w') as f:
        for t in failed:
            print(t, file=f)

main('../data/MUV/kinase/*')