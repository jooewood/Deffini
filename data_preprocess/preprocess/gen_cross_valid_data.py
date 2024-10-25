#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================
import os
import sys
import numpy as np
import random
import pickle
    
def read_list(file_path):
    f = open(file_path, 'r')
    task_folders = []
    content = f.readlines()
    for line in content:
        task_folders.append(line.split('\n')[0])
    random.shuffle(task_folders)
    return task_folders

def concat_train_file(file_path, out_folder):
    task_folders = read_list(file_path)
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
    
    file_name = file_path.split('/')[-1]
    set_name = file_name.split('.taskfolders')[0]
    path = '%s/%s.pickle' % (out_folder, set_name)
    with open(path, 'wb') as fb:
        pickle.dump([big_name, big_atm ,big_chrg, big_dist, big_amino,  \
            big_mask, big_y], fb, protocol=4)

def copy_test_files(file_path, test_dir, dataset):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    task_folders = read_list(file_path)
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        source = '{}/tmp/train.no_grid.pickle'.format(task_folder)
        if dataset=='kinformation':# kinformation dataset has 2 level dir
            target = task_folder.split('/')[-2]
            conf = task_folder.split('/')[-1]
            destination = '{}/{}.{}.pickle'.format(test_dir, target, conf)
            os.system('cp {} {}'.format(source, destination))
        elif dataset=="DUD-E":# DUD-E dataset has 1 level dir
            target = task_folder.split('/')[-1]
            destination = '{}/{}.pickle'.format(test_dir, target)
            os.system('cp {} {}'.format(source, destination))


            
def main(dataset, fold_number=3):
    # datgset = "DUD-E"
    root_path = '/y/Fernie/kinase_project'
    file_dir = os.path.join(root_path, 'cluster_3_fold', dataset, 'split_files')
    # global dataset
    working_dir = '/y/Fernie/scratch/cluster_3_fold'
    out_folder = os.path.join(working_dir, dataset, 'data')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i in range(fold_number):
        # i = 0
        file_path = '{}/train_{}.taskfolders'.format(file_dir, i)
        concat_train_file(file_path, out_folder)
        file_path = '{}/test_{}.taskfolders'.format(file_dir, i)
        test_dir = '{}/test_{}'.format(out_folder, i)
        copy_test_files(file_path, test_dir, dataset)


    
if __name__ == "__main__":
    main("DUD-E")
    # main("kinformation")
    # cluster 3 里的kinformation数据地址是
    # /home/tensorflow/kinase_project/data/kinformation/*
    # 暂时不可用
