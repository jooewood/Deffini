#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import sys
import glob
import pandas as pd
# data_root = '../../data/MUV/kinase'

def copy_files(origin, goal):
    if not os.path.exists(goal):
        os.makedirs(goal)
    os.system('cp %s %s' % (origin, goal))

def change_directory(data_root):
    new_folder = data_root + '_new'
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    task_folders = glob.glob(data_root+'/*')
    failed_list = []
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        target = task_folder.split('/')[-1]
        origin = task_folder + '/ligand_active/*.pdbqt'
        goal = new_folder + '/%s/cmpd_library/active' % target
        copy_files(origin, goal)
        origin = task_folder + '/ligand_decoy/*.pdbqt'
        goal = new_folder + '/%s/cmpd_library/decoy' % target
        copy_files(origin, goal)
        origin = task_folder + '/vs_active/*.pdbqt'
        goal = new_folder + '/%s/docking_pdbqt/active' % target
        copy_files(origin, goal)
        origin = task_folder + '/vs_decoy/*.pdbqt'
        goal = new_folder + '/%s/docking_pdbqt/decoy' % target
        copy_files(origin, goal)
        origin = task_folder + '/receptor/*'
        goal = new_folder + '/%s/target' % target
        copy_files(origin, goal)
        origin = task_folder + '/ligand_ref/*'
        goal = new_folder + '/%s/target' % target
        copy_files(origin, goal)
        origin = task_folder + '/scoring_smina/*'
        goal = new_folder + '/%s/scoring_smina' % target
        copy_files(origin, goal)
        
        path = task_folder + '/ranking/%s.ranking.smina_score.tsv' % target
        df = pd.read_table(path)
        path = task_folder + '/%s.actives.tsv' % target
        df_active = pd.read_table(path)
        path = task_folder + '/%s.decoys.tsv' % target
        df_decoy = pd.read_table(path)
        df_merge = pd.concat([df_active, df_decoy])
        df = pd.merge(df, df_merge, left_on='#Mol', right_on='Mol', how='inner')
        df = df.rename(columns={'Source':'Label', 'Conf':'#Cmpd_ID'})
        df['Label'] = df['Label'].replace(['ACTIVE', 'DECOY'], [1, 0])
        df = df[['#Cmpd_ID', 'Label']]
        path = new_folder + '/%s/answer' % target
        if not os.path.exists(path):
            os.makedirs(path)
        path = new_folder + '/%s/answer/docking.answer.tsv' % target
        df = df.sort_values(by='Label', ascending=False)
        df = df.reset_index(drop=True)
        path = new_folder + '/%s/answer/docking.answer.tsv' % target
        df.to_csv(path, sep='\t', index=False)

def extract_smina(data_root):
    task_folders = glob.glob(data_root+'/*')
    failed_list = []
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        smina_score_file = '{}/scoring_smina/docking.smina.tsv'.format(task_folder)
        if not os.path.exists('{}/scoring_smina'.format(task_folder)):
            os.makedirs('{}/scoring_smina'.format(task_folder))
        active_smina_score_files = glob.glob(task_folder+'/scoring_active/*')
        fb = open(smina_score_file, 'w')
        fb.write("#Pose_ID\tsmina_score\n")
        for file in active_smina_score_files:
            # file = active_smina_score_files[0]
            f = open(file, 'r')
            content = f.readlines()
            f.close()
            ID = file.split('/')[-1].split('.smina')[0]
            try:
                score = float(content[25].split()[1])
                print("%s\t%.4f" % (ID, score), file = fb)
            except:
                failed_list.append(file)
                
        decoy_smina_score_files = glob.glob(task_folder+'/scoring_decoy/*')
        for file in decoy_smina_score_files:
            # file = active_smina_score_files[0]
            f = open(file, 'r')
            content = f.readlines()
            f.close()
            ID = file.split('/')[-1].split('.smina')[0]
            try:
                score = float(content[25].split()[1])
                print("%s\t%.4f" % (ID, score), file = fb)
            except:
                failed_list.append(file)
        fb.close()

    for file in failed_list:
        print('{} failed to extract smina score'.format(file))
        f = open(file, 'r')
        print(f.read())
        f.close()
        
            
if __name__ == "__main__":
    extract_smina('../../data/MUV/kinase')
    extract_smina('../../data/MUV/non_kinase')
    change_directory('../../data/MUV/kinase')
    change_directory('../../data/MUV/non_kinase')
    