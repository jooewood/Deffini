#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Tensroflow 1.14.0
# @author: zdx
#input of train_main
#    path_to_save_model = '/home/zdx/Desktop/test/models/test.h5'
#    train_file = '/home/zdx/kinase_project/3_fold_cross_validation/kinformation/data/train_0.pickle'
#    test_dir = '/home/zdx/kinase_project/3_fold_cross_validation/kinformation/data/test_0' or None
#    BATCH_SIZE = 512
#    iteration = 5
# =============================================================================

import os
import sys
import glob
import shutil
import pickle
import random
import datetime
import argparse
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

from scipy import sparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, \
    Conv1D, Conv2D, MaxPooling2D, Layer, Reshape, Embedding
from tensorflow.keras import Input, Model
from tensorflow.keras.utils import multi_gpu_model, plot_model
#from sklearn.utils import class_weight
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

def get_mask_mat(mask):
    return np.array([np.vstack((np.ones((i, 1, cf)), \
        np.zeros((max_atom_num - i, 1, cf)))) for i in mask])

def DDK_model():
    raw_atm = Input(shape=(max_atom_num, kc+kp), dtype='int32', name='atm')
    raw_dist = Input(shape=(max_atom_num, kc+kp), dtype='int32', name='dist')
    raw_chrg = Input(shape=(max_atom_num, kc+kp), dtype='int32', name='chrg')
    raw_amino = Input(shape=(max_atom_num, kp), dtype='int32', name='amino')
    mask_mat = Input(shape=(max_atom_num, 1, cf), dtype='float32', name='mask')

    embedded_atm  = Embedding(A, datm, input_length=\
        max_atom_num * (kc + kp))(raw_atm)
    embedded_dist = Embedding(D, ddist, input_length=\
        max_atom_num * (kc + kp))(raw_dist)
    embedded_chrg = Embedding(C, dchrg, input_length=\
        max_atom_num * (kc + kp))(raw_chrg)
    embedded_amino = Embedding(R, damino, input_length=\
        max_atom_num * kp)(raw_amino)

    embedded_atm  = tf.reshape(embedded_atm, [-1, max_atom_num, \
        (kc + kp) * datm ])
    embedded_dist = tf.reshape(embedded_dist, [-1, max_atom_num, \
        (kc + kp) * datm ])
    embedded_chrg = tf.reshape(embedded_chrg, [-1, max_atom_num, \
        (kc + kp) * datm ])
    embedded_amino = tf.reshape(embedded_amino, [-1, max_atom_num, \
        kp * datm ])

    Z = tf.expand_dims(tf.concat([embedded_atm, embedded_dist, embedded_chrg, \
        embedded_amino], axis=-1), -1)

    input_shape = (100, 5200, 1)
    conv1 = Conv2D(cf, (1, zi), strides=1, input_shape=input_shape)(Z)
    conv1 = conv1 * mask_mat
    pool1 = MaxPooling2D(pool_size=(max_atom_num,1), strides=1)(conv1)
    pool1 = Reshape((1,cf,1))(pool1)

    conv2 = Conv2D(h,  (1, cf))(pool1)
    conv2 = Reshape((1,h,1))(conv2)

    conv3 = Conv2D(2,  (1, h))(conv2)
    flat1 = Flatten()(conv3)
    score = Dense(1, activation='sigmoid')(flat1)

    model = Model(inputs=[raw_atm, raw_dist, raw_chrg, raw_amino, mask_mat], \
        outputs=[score])
    model.summary()
    return model

def read_a_pickle(pickle_path):
    input_list = []
    with open(pickle_path, 'rb') as f:
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

def predict_one_target(pickle_path):
    Zinput, y, name_list = read_a_pickle(pickle_path)
    results = model.predict(Zinput, batch_size=512).flatten()
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
    return result_df

def score_each_target_in_testdir(test_dir, out_folder):
    path = test_dir + '/*.pickle'
    pickle_paths = glob.glob(path)
    failed = []
    auroc_list = []
    auprc_list = []
    number_of_auroc_over_ninety = 0
    for pickle_path in pickle_paths:
        # pickle_path = pickle_paths[0]
        try:
            df = predict_one_target(pickle_path)
            target = pickle_path.split('/')[-1]
            target = target.split('.pickle')[0]
            path = out_folder + '/{}.score'.format(target)
            df.to_csv(path, sep="\t", index=False)
            auroc, auprc = compute_area_under_roc_prc(df['label'], df['score'])
            if auroc > 0.9:
                number_of_auroc_over_ninety += 1
            auroc_list.append(auroc)
            auprc_list.append(auprc)
        except:
            failed.append(pickle_path)
            continue
    ave_auroc = round(np.mean(auroc_list), 4)
    ave_auprc = round(np.mean(auprc_list), 4)
    return failed, ave_auroc, ave_auprc, number_of_auroc_over_ninety
        

def train_main(path_to_save_model, train_file, test_dir=None, \
    BATCH_SIZE = 1024, iteration = 11):
    """
    path_to_save_model = '/home/zdx/Desktop/test/models/test.h5'
    train_file = '/home/zdx/kinase_project/3_fold_cross_validation/kinformation/data/train_0.pickle'
    test_dir = '/home/zdx/kinase_project/3_fold_cross_validation/kinformation/data/test_0'
    BATCH_SIZE = 512
    iteration = 5
    """
    ## train
    model_folder = os.path.dirname(path_to_save_model)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    starttime = datetime.datetime.now()
    global model
    model = DDK_model()# define model
    model.compile(loss = 'binary_crossentropy', 
            optimizer = 'adam', 
            metrics = ['accuracy'])
    with open(train_file, 'rb') as f: # load train data
        one_pickle = pickle.load(f)
    x = [one_pickle[1], one_pickle[2],  # model input
         one_pickle[3], one_pickle[4], 
         get_mask_mat(one_pickle[5])]
    y = one_pickle[6][:,1] # label
    ## start to train
    #for i in range(iteration): # train for muliti iterations       
    model.fit(x=x,y=y, batch_size = BATCH_SIZE, epochs=iteration, \
        shuffle=True, use_multiprocessing=True) # model fit
    ## save model
    model.save(path_to_save_model)
    ## compute total train time
    endtime = datetime.datetime.now()
    total_time = (endtime - starttime).seconds # get process time of one epoch

    if test_dir==None:
        return total_time
    else:
        del x,y 
        pre = os.path.dirname(os.path.dirname(path_to_save_model))
        out_folder = pre + '/scores'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        failed, ave_auroc, ave_auprc, number_of_auroc_over_ninety = \
            score_each_target_in_testdir(test_dir, out_folder)
        return total_time, failed, ave_auroc, ave_auprc, \
            number_of_auroc_over_ninety
