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
from pandas import DataFrame
import pandas as pd
import re

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

def score_with_one_model(model_path, task_folders, score_folder):
    global model
    failed = []
    model = load_model(model_path)
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        target_name = task_folder.split('/')[-1]
        try:
            df = predict_one_target(task_folder)
            score_path = '%s/%s.score' % (score_folder, target_name)
            df.to_csv(score_path, index=False)
        except:
            failed.append(model_path)
    K.clear_session()
    return failed
    

def score_with_mult_model(model_zoo, task_folders, score_zoo):
    # model_zoo = '../add_kinase/DUD-E_kinase/models'
    # task_folders = ''
    # score_zoo = '../add_kinase/DUD-E_kinase/MUV_scores'
    failed = []
    models_path = glob.glob(model_zoo+'/*.h5')
    for i in range(len(models_path)):
        # i = 0
        model_path = model_zoo +'/%d.h5' % (i+1)
        score_folder = score_zoo + '/%d' % (i+1)
        if not os.path.exists(score_folder):
            os.makedirs(score_folder)
        tmp_failed = score_with_one_model(model_path, task_folders, score_folder)
        failed = failed + tmp_failed
        
def score_one_dataset(targets_path, model_root):
    # targets_path = "../data/MUV/kinase"
    # model_root = '../add_kinase/DUD-E_kinase'
    m = re.search('data/(.+)/*', targets_path); m = m.group(1)
    if '/' in m:
        m = m.replace('/', '_')
    task_folders = glob.glob(targets_path+'/*')
    model_zoo = model_root + '/models'
    score_zoo = model_root + '/%s_scores' % m
    score_with_mult_model(model_zoo, task_folders, score_zoo)
    
def score_task_folders(name, task_folders, model_root):
    model_zoo = model_root + '/models'
    score_zoo = model_root + '/%s_scores' % name
    score_with_mult_model(model_zoo, task_folders, score_zoo)

def score_DUD_E_subfamily(name, model_root):
    # name = "kinase"
    # model_root = '../add_kinase/DUD-E_non_kinase'
    df = pd.read_csv('../data/DUD-E_summary.csv')
    df['Target Name'] = df['Target Name'].apply(lambda x: x.lower())
    df['family'] = df['family'].apply(lambda x: x.lower())
    df = df[df['family']==name]
    targets = list(df['Target Name'])
    task_folders = list(map(lambda x: '../data/DUD-E/'+x, targets))
    name = "DUD-E_" + name
    score_task_folders(name, task_folders, model_root)
    
# score_one_dataset("../data/MUV/kinase", '../add_kinase/DUD-E_kinase')
# score_one_dataset("../data/MUV/kinase", '../add_kinase/DUD-E_kinase_DUD-E_non_kinase')
# score_one_dataset("../data/MUV/kinase", '../add_kinase/DUD-E_non_kinase')
# score_one_dataset("../data/MUV/kinase", '../add_kinase/DUD-E_non_kinase_DUD-E_kinase')
# score_one_dataset("../data/MUV/kinase", '../add_kinase/KCD')
# score_one_dataset("../data/MUV/kinase", '../add_kinase/KCD_DUD-E_non_kinase')


score_DUD_E_subfamily('kinase', '../add_kinase/DUD-E_non_kinase')
score_DUD_E_subfamily('kinase', '../add_kinase/KCD')
score_DUD_E_subfamily('kinase', '../add_kinase/KCD_DUD-E_non_kinase')