#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

"""
Undersample test data for validation
"""
from tqdm import tqdm
from under_sampling import under_sample_a_target

for dataset in tqdm(['Kernie', 'DUD-E']):
    for fold in tqdm(range(3)):
        under_sample_a_target(f'/y/Aurora/Fernie/data/structure_based_data/{dataset}_3_fold/test_{fold}.pickle',
                              f'/y/Aurora/Fernie/data/structure_based_data/{dataset}_3_fold/test_{fold}.pickle')



under_sample_a_target('/y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_2.pickle',
                      '/y/Aurora/Fernie/data/structure_based_data/Kernie_3_fold/test_2.pickle')


"""
Copy all score file from /y/Aurora/Fernie/data/structure_based_data/Kernie
to  /y/Aurora/Fernie/output/Kernie/Smina/scores

"""
import os
import pandas as pd
from glob import glob
src_score_files = glob('/y/Aurora/Fernie/data/structure_based_data/Kernie/*/*/scoring_smina/docking.smina.tsv')
output_dir = '/y/Aurora/Fernie/output/Kernie/Smina/scores'
for src in tqdm(src_score_files):
    # src = src_score_files[0]
    df = pd.read_csv(src, sep='\t')
    target, con = src.split('/')[7:9]
    label = pd.read_csv(os.path.join(
        '/y/Aurora/Fernie/data/structure_based_data/Kernie', 
        target, con, 'answer', 'docking.answer.tsv'), sep='\t')
    df_m = df.merge(label, left_on='#Pose_ID', right_on='#Cmpd_ID')
    del df_m['#Cmpd_ID']
    df_m.columns = ['ID', 'score', 'label']
    df_m.score = -df_m.score
    df_m.sort_values('score', ascending = False, inplace=True)
    dst = os.path.join(output_dir, f'{target}.{con}.score')
    df_m.to_csv(dst, sep='\t')

src_score_files = glob('/y/Aurora/Fernie/data/structure_based_data/DUD-E/*/scoring_smina/docking.smina.tsv')
output_dir = '/y/Aurora/Fernie/output/DUD-E/Smina/scores'
for src in tqdm(src_score_files):
    # src = src_score_files[0]
    df = pd.read_csv(src, sep='\t')
    target = src.split('/')[7]
    label = pd.read_csv(os.path.join(
        '/y/Aurora/Fernie/data/structure_based_data/DUD-E', 
        target, 'answer', 'docking.answer.tsv'), sep='\t')
    df_m = df.merge(label, left_on='#Pose_ID', right_on='#Cmpd_ID')
    del df_m['#Cmpd_ID']
    df_m.columns = ['ID', 'score', 'label']
    df_m.score = -df_m.score
    df_m.sort_values('score', ascending = False, inplace=True)
    dst = os.path.join(output_dir, f'{target}.score')
    df_m.to_csv(dst, sep='\t')
    
"""
Copy all score file from /y/Aurora/Fernie/data/structure_based_data/MUV
to  /y/Aurora/Fernie/output/MUV/scores/Smina

"""
dataset = "MUV"
src_score_files = glob(f'/y/Aurora/Fernie/data/structure_based_data/{dataset}/*/*/scoring_smina/docking.smina.tsv')
output_dir = '/y/Aurora/Fernie/output/MUV/scores/Smina'
for src in tqdm(src_score_files):
    # src = src_score_files[0]
    df = pd.read_csv(src, sep='\t')
    family, target = src.split('/')[7:9]
    label = pd.read_csv(os.path.join(
        f'/y/Aurora/Fernie/data/structure_based_data/{dataset}', 
        family, target, 'answer', 'docking.answer.tsv'), sep='\t')
    df_m = df.merge(label, left_on='#Pose_ID', right_on='#Cmpd_ID')
    del df_m['#Cmpd_ID']
    df_m.columns = ['ID', 'score', 'label']
    df_m.score = -df_m.score
    df_m.sort_values('score', ascending = False, inplace=True)
    dst = os.path.join(output_dir, f'{target}.{family}.Smina.None.score')
    df_m.to_csv(dst, sep='\t')

src_score_files = glob('/y/Aurora/Fernie/data/structure_based_data/DUD-E/*/scoring_smina/docking.smina.tsv')
output_dir = '/y/Aurora/Fernie/output/DUD-E/Smina/scores'
for src in tqdm(src_score_files):
    # src = src_score_files[0]
    df = pd.read_csv(src, sep='\t')
    target = src.split('/')[7]
    label = pd.read_csv(os.path.join(
        '/y/Aurora/Fernie/data/structure_based_data/DUD-E', 
        target, 'answer', 'docking.answer.tsv'), sep='\t')
    df_m = df.merge(label, left_on='#Pose_ID', right_on='#Cmpd_ID')
    del df_m['#Cmpd_ID']
    df_m.columns = ['ID', 'score', 'label']
    df_m.score = -df_m.score
    df_m.sort_values('score', ascending = False, inplace=True)
    dst = os.path.join(output_dir, f'{target}.score')
    df_m.to_csv(dst, sep='\t')



"""
Adjust incorrect MUV file names
"""
import os
import pandas as pd
from utils import try_copy
from tqdm import tqdm 

input_dir = '/y/Aurora/Fernie/output/MUV/scores/old_lf_transformer'
model_name = 'transformer'
output_dir = '/y/Aurora/Fernie/output/MUV/scores/transformer'
dict_ = {
    'all_DUDE':'DUD-E',
    'all_Kernie': 'Kernie',
    'kinase_only': 'DUD-E_kinase',
    'all-muv': 'Kernie-MUV',
    'protease_only': 'DUD-E_protease',
    }

def adjust_MUV_score_filename(input_dir, output_dir, model_name):
    MUV_info = pd.read_excel('/y/Aurora/Fernie/data/MUV_info.xlsx')
    MUV_info = MUV_info[['confirm. assay (AID)', 'family']]
    MUV_info['confirm. assay (AID)'] = [x.strip() for x in MUV_info['confirm. assay (AID)']]
    MUV_info.set_index('confirm. assay (AID)', inplace=True)
    MUV_info = MUV_info.to_dict()['family']
    file_names = os.listdir(input_dir)
    for file in tqdm(file_names):
        # file = file_names[1]
        target_name, training_set, method = file.split('.')[0:3]
        if target_name=='689':
            target_family = MUV_info[target_name+'b']
        else:
            target_family = MUV_info[target_name]
        ad_training_set = dict_[training_set]
        new_name = '.'.join([target_name, target_family, model_name, ad_training_set, 'score'])
        try_copy(os.path.join(input_dir, file), os.path.join(output_dir, new_name))