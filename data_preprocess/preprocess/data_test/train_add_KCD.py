#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""


import pandas as pd
import numpy as np
import glob
import os
import random
import pickle
import train
import time
def integrate_feature(task_folders):
    name_list     = []
    atm_list      = []
    chrg_list     = []
    dist_list     = []
    amino_list    = []
    mask_mat_list = []
    y_list        = []
    for task_folder in task_folders:
        path = '%s/tmp/feature_undersample.pickle' % task_folder
        with open(path, 'rb') as fb:
            one_target_feature = pickle.load(fb)
        name_list.append(one_target_feature[0])
        atm_list.append(one_target_feature[1])
        chrg_list.append(one_target_feature[2])
        dist_list.append(one_target_feature[3])
        amino_list.append(one_target_feature[4])
        mask_mat_list.append(one_target_feature[5])
        y_list.append(one_target_feature[6])
        
    big_name  = np.concatenate(name_list, axis=0)
    big_atm   = np.concatenate(atm_list, axis=0)
    big_chrg  = np.concatenate(chrg_list, axis=0)
    big_dist  = np.concatenate(dist_list, axis=0)
    big_amino = np.concatenate(amino_list, axis=0)
    big_mask  = np.concatenate(mask_mat_list, axis=0)
    big_y     = np.concatenate(y_list, axis=0)
    return [big_name, big_atm ,big_chrg, big_dist, big_amino,  big_mask, big_y]

def remain_items_that_are_dir(x):
	def judge_whether_is_dir(path):
		if not os.path.isdir(path):
			return False
		else:
			return True
	return list(filter(judge_whether_is_dir, x))

def extract(data, whole_data): # extract data from 'whole_data' according to 'data'
    extract_data = []
    for i in data:
        for j in whole_data:
            if i in j:
                extract_data.append(j)
    return extract_data

def main():
    dataset = "KCD_358"
    folder = "../add_kinase/%s" % dataset
    KCD_task_folders = glob.glob('../data/kinformation/*/*')
    KCD_task_folders = remain_items_that_are_dir(KCD_task_folders)
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder+'/models')
        os.makedirs(folder+'/MUV_scores')
        os.makedirs(folder+'/features')
    targets_path = '%s/targets.pickle' % folder
    if not os.path.exists(targets_path):
        targets = os.listdir('../data/kinformation/')
        random.shuffle(targets)
        with open(targets_path, 'wb') as fb:
            pickle.dump(targets, fb)
    else:
        with open(targets_path, 'rb') as fb:
            targets = pickle.load(fb)
    path = folder
    for j in range(18, 0, -1):
        # j = 26
        cur_targets = targets[0:j]
        task_folders = extract(cur_targets, KCD_task_folders)
        big_feature = integrate_feature(task_folders)
        train_file = '%s/features/tmp.pickle' % (path)
        with open(train_file, 'wb') as fb:
            pickle.dump(big_feature, fb)
        path_to_save_model = '%s/models/%d.h5' % (path, j)
        train.train_main(path_to_save_model, train_file, test_dir=None, BATCH_SIZE = 1024, iteration = 11)
main()
