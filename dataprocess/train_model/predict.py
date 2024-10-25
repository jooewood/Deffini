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

now_dir = os.getcwd()
prefix = os.path.dirname(os.path.dirname(now_dir))

def compute_area_under_roc_prc(y_true, y_pred): # need to be rank first
    auc_roc, auc_prc = np.nan, np.nan
    a_pos = []
    for i, y_tmp in enumerate(y_true):
        if y_tmp==1:
            a_pos.append(i+1)
    a_pos = np.array(a_pos)
    a_seen = np.array(range(1,len(a_pos)+1))
    d_seen = a_pos - a_seen
    ##Calculate AUC ROC
    a_total = len(a_pos)
    d_total = len(y_true) - len(a_pos)
    contri_roc = d_seen/(a_total*d_total)
    auc_roc = 1.0 - np.sum(contri_roc)
    ##Calculate AUC PRC
    auc_prc = (1/a_total) * np.sum((1.0*a_seen)/(a_seen+d_seen))
    return round(auc_roc,4), round(auc_prc,4)

def read_list(file_path):
    f = open(file_path, 'r')
    task_folders = []
    content = f.readlines()
    for line in content:
        task_folders.append(line.split('\n')[0])
    return task_folders

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

def predict_one_target(task_folder, cmpd_lib):
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

def batch_predict(task_folders):
    failed = []
    auc_roc_list = []
    auc_prc_list = []
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        try:
            df = predict_one_target(task_folder, cmpd_lib=None)
            y_pred = list(df['score'])
            y_true = list(df['label'])
            auc_roc, auc_prc = compute_area_under_roc_prc(y_true, y_pred)
            auc_roc_list.append(auc_roc)
            auc_prc_list.append(auc_prc)
        except:
            failed.append(task_folder)
    ave_auc_roc = round(np.mean(auc_roc_list),4)
    ave_auc_prc = round(np.mean(auc_prc_list),4)
    return ave_auc_roc, ave_auc_prc

def predict_main(model_path, file_path):
    # model_path = "/home/zdx/kinase_project/model/kinformation201907311703.h5"
    # file_path = "/home/zdx/kinase_project/cross_validation/split_0/split_files/test_0.tsv"
    global model
    model = load_model(model_path)
    task_folders = read_list(file_path)
    # task_folders = task_folders[0:2]
    ave_auc_roc, ave_auc_prc = batch_predict(task_folders)
    return ave_auc_roc, ave_auc_prc

def test_fn(model_path, task_folder, cmpd_lib=None):
    global model
    model = load_model(model_path)
    df = predict_one_target(task_folder, cmpd_lib)
    return df

def main(task_folder, model_path, cmpd_lib=None):
    # model_path = "/home/zdx/kinase_project/model/kinformation_20190801_whole.h5"
    # task_folder = "/home/zdx/kinase_project/benchmark_data/1.publications/d4_nature"
    # cmpd_lib = "Nature_548_SINGLE"
    global model
    model = load_model(model_path)
    df = predict_one_target(task_folder, cmpd_lib)
    result_folder = "{}/scoring_DDK".format(task_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    path = "{0}/{1}.DDK.tsv".format(result_folder, cmpd_lib)
    df.to_csv(path, sep='\t', index=False)

#if __name__ == "__main__":
#    argv = sys.argv[1:]
#    task_folder = argv[0]
#    cmpd_lib = argv[1]
#    main(task_folder, cmpd_lib)
