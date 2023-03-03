#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle

def extract(data, whole_data): # extract data from 'whole_data' according to 'data'
    extract_data = []
    for i in data:
        for j in whole_data:
            if i in j:
                extract_data.append(j)
    return extract_data


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

def main(dataset):
    # dataset = "DUD-E_protease"
    root_path = '/y/Fernie/kinase_project'
    data_root = os.path.join(root_path, 'data', dataset)
    df = pd.read_csv(os.path.join(root_path, 'data','DUD-E_summary.csv'))
    if dataset=="DUD-E":
        task_folders = glob.glob(data_root + '/*')
    elif dataset=="kinformation":
        task_folders = glob.glob(data_root+ '/*/*')
    elif dataset=="DUD-E_non_kin":
        
        df['Target Name'] = df['Target Name'].apply(lambda x: x.lower())
        df = df.query('family!="Kinase"')
        task_folders = list(map(lambda x: os.path.join(\
            root_path, 'data','DUD-E',x), \
            list(df['Target Name'])))

    elif dataset=="DUD-E_kinase":
        df['Target Name'] = df['Target Name'].apply(lambda x: x.lower())
        df = df.query('family=="Kinase"')
        task_folders = list(map(lambda x: os.path.join(\
            root_path, 'data','DUD-E',x), \
            list(df['Target Name'])))

    elif dataset=="DUD-E_protease":
        df['Target Name'] = df['Target Name'].apply(lambda x: x.lower())
        df = df.query('family=="Protease"')
        task_folders = list(map(lambda x: os.path.join(\
            root_path, 'data','DUD-E',x), \
            list(df['Target Name'])))

    elif dataset=="DUD-E_non_protease":
        df['Target Name'] = df['Target Name'].apply(lambda x: x.lower())
        df = df.query('family!="Protease"')
        task_folders = list(map(lambda x: os.path.join(\
            root_path, 'data','DUD-E',x), \
            list(df['Target Name'])))

    elif dataset=="KCD-644":
        dataset = "kinformation"
        data_root = os.path.join(root_path, 'data', dataset)
        task_folders = glob.glob(data_root+ '/*/*')
        def myFunc(x):
            if 'O75116' in x:
                return False
            else:
                return True
        task_folders  = list(filter(myFunc, task_folders))
        dataset = "KCD-644"

    elif dataset=="KCD-548":
        dataset = "kinformation"
        data_root = os.path.join(root_path, 'data', dataset)
        task_folders = glob.glob(data_root+ '/*/*')
        def myFunc(x):
            if 'P17612' in x or 'P61925' in x:
                return False
            else:
                return True
        task_folders  = list(filter(myFunc, task_folders))
        dataset = "KCD-548"

    elif dataset=="KCD-810":
        dataset = "kinformation"
        data_root = os.path.join(root_path, 'data', dataset)
        task_folders = glob.glob(data_root+ '/*/*')
        def myFunc(x):
            if 'Q05397' in x:
                return False
            else:
                return True
        task_folders  = list(filter(myFunc, task_folders))
        dataset = "KCD-810"

    elif dataset=="KCD-689":
        dataset = "kinformation"
        data_root = os.path.join(root_path, 'data', dataset)
        task_folders = glob.glob(data_root+ '/*/*')
        def myFunc(x):
            if 'Q03137' in x:
                return False
            else:
                return True
        task_folders  = list(filter(myFunc, task_folders))
        dataset = "KCD-689"

    elif dataset=="KCD-MUV":
        dataset = "kinformation"
        data_root = os.path.join(root_path, 'data', dataset)
        task_folders = glob.glob(data_root+ '/*/*')
        def myFunc(x):
            if 'O75116' in x or 'Q03137' in x or 'P17612' in x or \
                'P61925' in x or 'Q05397' in x:
                return False
            else:
                return True
        task_folders  = list(filter(myFunc, task_folders))
        dataset = "KCD-MUV"

    big_feature = integrate_feature(task_folders)
    working_dir = '/y/Fernie/scratch'
    out_folder = os.path.join(working_dir,'final_model')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    path = '%s/%s.pickle' % (out_folder, dataset)
    with open(path, 'wb') as fb:
        pickle.dump(big_feature, fb, protocol=4)

if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")
    main("DUD-E_non_kin")
    main("DUD-E_kinase")
    main("KCD-644")
    main("KCD-548")
    main("KCD-810")
    main("KCD-689")
    main("KCD-MUV")
    main("DUD-E_protease")
    main("DUD-E_non_protease")
