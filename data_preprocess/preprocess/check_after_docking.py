#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import glob
import pandas as pd

def get_data(path, dataset):

    targets = []
    dock_lack_list = []
    dock_null_list = []
    cmpd_num = []
    dock_num = []

#     task_folders = glob.glob(path)
    task_folders = glob.glob(os.path.join(path,'*'))
    for task_folder in task_folders:
        if pro_dataset=='DUD-E':
            sub_folds = glob.glob(task_folder)
        elif pro_dataset=='kinformation':
            sub_folds = glob.glob(task_folder + '/*')
        else:
            sub_folds = glob.glob(task_folder)        
        
        for i in sub_folds:
            targets.append(i.split('/')[-1]) # get targets
#             cmpd_paths = glob.glob(i+'/cmpd_library/{}/*'.format(dataset)) 
#             dock_paths = glob.glob(i+'/docking_pdbqt/{}/*'.format(dataset))
            # get files' path in cmpd_library
            cmpd_paths = glob.glob(os.path.join(\
                i+'/cmpd_library/{}/'.format(dataset),'*')) 
            # get files' path in docking_pdbqt
            dock_paths = glob.glob(os.path.join(\
                i+'/docking_pdbqt/{}/'.format(dataset),'*'))
 
            cmpd_n = len(cmpd_paths) # get file number in cmpd_library
            dock_n = len(dock_paths) # get file number in docking_pdbqt
            cmpd_num.append(cmpd_n) # add the cmpd files' number into list
            dock_num.append(dock_n) # add the dock files' number into list
            dock_lack = cmpd_n - dock_n # compuete difference value between cmpd and dock
            dock_lack_list.append(dock_lack)
            dock_null = 0
            for j in dock_paths:
                if not os.path.getsize(j):
                    dock_null += 1
            dock_null_list.append(dock_null)
    return targets, dock_lack_list, dock_null_list, cmpd_num, dock_num

def check_after_docking(path):
    datasets = ['active', 'decoy']
    df_list = []
    for dataset in datasets:
        targets, dock_lack_list, dock_null_list, cmpd_num, dock_num = \
            get_data(path, dataset)
        df_t = pd.DataFrame({'targets': targets, 
                           '%s_dock_lack'%dataset: dock_lack_list, 
                           '%s_dock_null'%dataset: dock_null_list,
                           '%s_cmpd_num'%dataset: cmpd_num,
                           '%s_dock_num'%dataset: dock_num,
                          })
        df_list.append(df_t)
    df = pd.merge(df_list[0], df_list[1], on = 'targets')
    df['decoy_minus_active'] = df['decoy_dock_num'] - df['active_dock_num']
    return df


working_dir = '/y/Fernie/scratch'
path = '/y/Fernie/kinase_project/data/DUD-E/'
if path[-1] == '/':
    path = path[:-1]
pro_dataset =  os.path.basename(path)

df = check_after_docking(path)
df.to_csv(os.path.join(working_dir,'check_kinase_after_docking.csv'), \
    index=False)
