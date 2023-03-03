#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import sys
import pandas as pd
from tqdm import tqdm
import os
from rdkit_utils import SMILES2mol2InChI
from shutil import copy

def try_copy(source, target):
    try:
       copy(source, target)
    except IOError as e:
       print("Unable to copy file. %s" % e)
       exit(1)
    except:
       print("Unexpected error:", sys.exc_info())
       exit(1)



# dataset  = 'DUD-E'

# input_dir = f'/y/Aurora/Fernie/data/{dataset}_clur_3_fold'
# output_root = '/y/Aurora/Fernie/data/structure_based_data/smiles_seq'
# data_root = '/y/Aurora/Fernie/data/ligand_based_data'

# dataset_root = os.path.join(data_root, dataset)
# for i in tqdm(range(3)):
#     # i = 0
#     set_ = 'train'
#     print(set_)
#     with open(os.path.join(input_dir, f'{set_}_{i}.targets'), 'r') as f:
#         targets = f.read().splitlines()
#     dfs = []
#     undersample_dfs = []
#     for target in tqdm(targets):
#         print(target)
#         # target = targets[0]
#         src = os.path.join(dataset_root, f'{target}.csv')
#         df = pd.read_csv(src, sep='\t')
#         df = SMILES2mol2InChI(df)
#         dfs.append(df)
#         active_n = len(df.query('label==1'))
#         active_df = df.query('label==1')
#         sampled_decoy = df.query('label==0').sample(active_n)
#         undersample_df = pd.concat([active_df, sampled_decoy])
#         undersample_dfs.append(undersample_df)
#     df_m = pd.concat(dfs)
#     df_m.drop_duplicates('InChI', inplace=True)
#     tmp_out = os.path.join(output_root, dataset, f'{set_}_{i}.csv')
#     df_m.to_csv(tmp_out, sep='\t', index=False)
    
#     df_undersample = pd.concat(undersample_dfs)
#     df_undersample.drop_duplicates('InChI', inplace=True)
#     tmp_out = os.path.join(output_root, dataset, f'{set_}_{i}.undersample.csv')
#     df_undersample.to_csv(tmp_out, sep='\t', index=False)

#     set_ = 'test'
#     print(set_)
#     tmp_out = os.path.join(output_root, dataset, f'{set_}_{i}')
#     if not os.path.exists(tmp_out):
#         os.makedirs(tmp_out)
#     with open(os.path.join(input_dir, f'{set_}_{i}.targets'), 'r') as f:
#         targets = f.read().splitlines()
#     for target in tqdm(targets):
#         print(target)
#         src = os.path.join(dataset_root, f'{target}.csv')
#         df = pd.read_csv(src, sep='\t')
#         df = SMILES2mol2InChI(df)
#         df.drop_duplicates('InChI', inplace=True)
#         dst = os.path.join(tmp_out, f'{target}.csv')
#         df.to_csv(dst, sep='\t', index=False)

# dataset  = 'Kernie'

# input_dir = f'/y/Aurora/Fernie/data/{dataset}_clur_3_fold'
# output_root = '/y/Aurora/Fernie/data/structure_based_data/smiles_seq'
# data_root = '/y/Aurora/Fernie/data/ligand_based_data'

# dataset_root = os.path.join(data_root, dataset)
# for i in tqdm(range(3)):
#     # i = 0
#     set_ = 'train'
#     print(set_)
#     with open(os.path.join(input_dir, f'{set_}_{i}.targets'), 'r') as f:
#         targets = f.read().splitlines()
#     dfs = []
#     undersample_dfs = []
#     for target in tqdm(targets):
#         print(target)
#         # target = targets[0]
#         src = os.path.join(dataset_root, f'{target}.csv')
#         df = pd.read_csv(src, sep='\t')
#         df = SMILES2mol2InChI(df)
#         dfs.append(df)
#         active_n = len(df.query('label==1'))
#         active_df = df.query('label==1')
#         sampled_decoy = df.query('label==0').sample(active_n)
#         undersample_df = pd.concat([active_df, sampled_decoy])
#         undersample_dfs.append(undersample_df)
#     df_m = pd.concat(dfs)
#     df_m.drop_duplicates('InChI', inplace=True)
#     tmp_out = os.path.join(output_root, dataset, f'{set_}_{i}.csv')
#     df_m.to_csv(tmp_out, sep='\t', index=False)
    
#     df_undersample = pd.concat(undersample_dfs)
#     df_undersample.drop_duplicates('InChI', inplace=True)
#     tmp_out = os.path.join(output_root, dataset, f'{set_}_{i}.undersample.csv')
#     df_undersample.to_csv(tmp_out, sep='\t', index=False)

#     set_ = 'test'
#     print(set_)
#     tmp_out = os.path.join(output_root, dataset, f'{set_}_{i}')
#     if not os.path.exists(tmp_out):
#         os.makedirs(tmp_out)
#     with open(os.path.join(input_dir, f'{set_}_{i}.targets'), 'r') as f:
#         targets = f.read().splitlines()
#     for target in tqdm(targets):
#         print(target)
#         src = os.path.join(dataset_root, f'{target}.csv')
#         df = pd.read_csv(src, sep='\t')
#         df = SMILES2mol2InChI(df)
#         df.drop_duplicates('InChI', inplace=True)
#         dst = os.path.join(tmp_out, f'{target}.csv')
#         df.to_csv(dst, sep='\t', index=False)

# def concat_according_to_df(dataset, sub_info, dataset_root, family, output_root):
#     sub_targets = [x.lower() for x in sub_info['Target Name']]
#     dfs = []
#     undersample_dfs = []
#     for target in tqdm(sub_targets):
#         # target = sub_targets[0]
#         src = os.path.join(dataset_root, f'{target}.csv')
#         df = pd.read_csv(src, sep='\t')
#         df = SMILES2mol2InChI(df)
#         dfs.append(df)
#         active_n = len(df.query('label==1'))
#         active_df = df.query('label==1')
#         sampled_decoy = df.query('label==0').sample(active_n)
#         undersample_df = pd.concat([active_df, sampled_decoy])
#         undersample_dfs.append(undersample_df)
#     df_m = pd.concat(dfs)
#     df_m.drop_duplicates('InChI', inplace=True)
#     tmp_out = os.path.join(output_root, dataset, f'{family}.csv')
#     df_m.to_csv(tmp_out, sep='\t', index=False)
    
#     df_undersample = pd.concat(undersample_dfs)
#     df_undersample.drop_duplicates('InChI', inplace=True)
#     tmp_out = os.path.join(output_root, dataset, f'{family}.undersample.csv')
#     df_undersample.to_csv(tmp_out, sep='\t', index=False)

# dataset = 'DUD-E'
# info_path = '/y/Aurora/Fernie/data/DUD-E_info.csv'
# info = pd.read_csv(info_path)
# familys = set(info['family'])
# data_root = '/y/Aurora/Fernie/data/ligand_based_data'
# output_root = '/y/Aurora/Fernie/data/structure_based_data/smiles_seq'
# dataset_root = os.path.join(data_root, dataset)
# for family in tqdm(familys):
#     # family = "Kinase"
#     print(dataset, family, '\n')
#     sub_info = info.query("family=='%s'" % family)
#     concat_according_to_df(dataset, sub_info, dataset_root, family, output_root)

def concat_dfs(targets, dataset_root, output_root, out_name):
    dfs = []
    undersample_dfs = []
    for target in tqdm(targets):
        src = os.path.join(dataset_root, f'{target}.csv')
        df = pd.read_csv(src, sep='\t')
        df = SMILES2mol2InChI(df)
        dfs.append(df)
        active_n = len(df.query('label==1'))
        active_df = df.query('label==1')
        sampled_decoy = df.query('label==0').sample(active_n)
        undersample_df = pd.concat([active_df, sampled_decoy])
        undersample_dfs.append(undersample_df)
    df = pd.concat(dfs)
    df.drop_duplicates('InChI', inplace=True)
    out = os.path.join(output_root, f'{out_name}.csv')
    df.to_csv(out, sep='\t', index=False)
    
    df = pd.concat(undersample_dfs)
    df.drop_duplicates('InChI', inplace=True)
    out = os.path.join(output_root, f'{out_name}.undersample.csv')
    df.to_csv(out, sep='\t', index=False)

"""
whole DUD-E
"""

# targets = os.listdir('/y/Aurora/Fernie/data/structure_based_data/DUD-E')

# concat_dfs(targets, '/y/Aurora/Fernie/data/ligand_based_data/DUD-E',
#             '/y/Aurora/Fernie/data/structure_based_data/smiles_seq/DUD-E',
#             out_name='DUD-E')

"""
whole Kernie
"""
print('whole Kernie')

with open('/y/Aurora/Fernie/data/Kernie.targets', 'r') as f:
    targets = f.read().splitlines()
concat_dfs(targets, '/y/Aurora/Fernie/data/ligand_based_data/Kernie',
           '/y/Aurora/Fernie/data/structure_based_data/smiles_seq/Kernie',
           out_name='Kernie')

for x in targets:
    if 'P17612' in x or 'P61925' in x or 'O75116' in x or 'Q03137' in x or\
       'Q05397' in x:
        targets.remove(x)
        
concat_dfs(targets, '/y/Aurora/Fernie/data/ligand_based_data/Kernie',
           '/y/Aurora/Fernie/data/structure_based_data/smiles_seq/Kernie',
           out_name='Kernie-MUV')