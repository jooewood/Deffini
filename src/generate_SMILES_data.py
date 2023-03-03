xi#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import pandas as pd
from rdkit.Chem import PandasTools
from tqdm import tqdm
import glob
import rdkit
from rdkit import Chem
import numpy as np

"""
-------------------------------------------------------------------------------
DUD-E
-------------------------------------------------------------------------------
"""

input_dir = '/y/Aurora/Fernie/data/structure_based_data/raw/DUD-E'
output_dir = '/y/Aurora/Fernie/data/ligand_based_data/DUD-E'

failed = []
target_names = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
for target_name in tqdm(target_names):
    try:
        active_sdf = f'{input_dir}/{target_name}/actives_final.sdf.gz'
        decoy_sdf = f'{input_dir}/{target_name}/decoys_final.sdf.gz'
        active_df = PandasTools.LoadSDF(active_sdf,smilesName='SMILES', idName='ID')
        del active_df['ROMol']
        active_df['label'] = 1
        decoy_df = PandasTools.LoadSDF(decoy_sdf,smilesName='SMILES', idName='ID')
        del decoy_df['ROMol']
        decoy_df['label'] = 0
        df = pd.concat([active_df, decoy_df])
        df.to_csv(os.path.join(output_dir, f'{target_name}.csv'), sep='\t', index=False)
    except:
        failed.append(target_name)
        continue

print(failed)

"""
-------------------------------------------------------------------------------
Kernie
-------------------------------------------------------------------------------
"""
# output_dir = '/y/Aurora/Fernie/data/ligand_based_data/Kernie'

# data_root = '/y/Aurora/Fernie/kinase_project/data/kinformation/'
# target_file = '/y/Aurora/Fernie/data/structure_based_data/Kernie.targets'
# with open(target_file, 'r') as f:
#     data = f.readlines()

# # file = '/y/Aurora/Fernie/Kernie/A3FQ16/codi/cmpd_library/active.txt'

# def get_ID(file):
#     with open(file, 'r') as f:
#         data = f.read().splitlines()
#         data = data[1:]
#     return data


# for task_folder in tqdm(data):
#     task_folder = task_folder.strip()
#     active_sdf = os.path.join(data_root, task_folder, 'cmpd_library/active.sdf')
#     decoy_sdf = os.path.join(data_root, task_folder, 'cmpd_library/decoy.sdf')
#     active_txt = os.path.join(data_root, task_folder, 'cmpd_library/active.txt')
#     decoy_txt = os.path.join(data_root, task_folder, 'cmpd_library/decoy.txt')
    
#     active_df = PandasTools.LoadSDF(active_sdf,smilesName='SMILES', idName='ID')
#     del active_df['ROMol']
#     active_df['label'] = 1
#     active_df['ID'] = get_ID(active_txt)
#     decoy_df = PandasTools.LoadSDF(decoy_sdf,smilesName='SMILES', idName='ID')
#     del decoy_df['ROMol']
#     decoy_df['label'] = 0
#     decoy_df['ID'] = get_ID(decoy_txt)
#     df = pd.concat([active_df, decoy_df])
#     task_folder = task_folder.replace('/', '.')
#     df.to_csv(os.path.join(output_dir, f'{task_folder}.csv'), sep='\t', index=False)

"""
-------------------------------------------------------------------------------
MUV
-------------------------------------------------------------------------------
"""
input_dir = '/y/Aurora/Fernie/data/structure_based_data/MUV'
output_dir = '/y/Aurora/Fernie/data/ligand_based_data/MUV'

task_folders = glob.glob(os.path.join(input_dir, '*', '*'))


def pdbqt2pdb(line):
    try:
        os.system(' '.join(['babel','-ipdbqt', line['pdbqt_path'], '-opdb', line['pdb_path']]))
        mol = rdkit.Chem.rdmolfiles.MolFromPDBFile(line['pdb_path'])
        return Chem.MolToSmiles(mol)
    except:
        return np.nan

def pdbqtpath2pdbpath(path):
    return path.replace('.pdbqt', '.pdb')
def get_ID(path):
    return os.path.basename(path).split('.pdbqt')[0]

for task_folder in tqdm(task_folders):
    # task_folder = task_folders[0]
    active_dir = os.path.join(task_folder, 'cmpd_library/active')
    decoy_dir = os.path.join(task_folder, 'cmpd_library/decoy')
    active_pdbqt_paths = glob.glob(os.path.join(active_dir, '*.pdbqt'))
    decoy_pdbqt_paths = glob.glob(os.path.join(decoy_dir, '*.pdbqt'))
    active_pdb_paths = list(map(pdbqtpath2pdbpath, active_pdbqt_paths))
    decoy_pdb_paths = list(map(pdbqtpath2pdbpath, decoy_pdbqt_paths))
    active_IDs = list(map(get_ID, active_pdbqt_paths))
    decoy_IDs = list(map(get_ID, decoy_pdbqt_paths))
    active_df = pd.DataFrame({
        'ID': active_IDs,
        'pdbqt_path':active_pdbqt_paths,
        'pdb_path':active_pdb_paths,
        'label': 1
        })
    decoy_df = pd.DataFrame({
        'ID': decoy_IDs,
        'pdbqt_path':decoy_pdbqt_paths,
        'pdb_path':decoy_pdb_paths,
        'label': 0
        })
    df = pd.concat([active_df, decoy_df])
    family = task_folder.split('/')[-2]
    target = task_folder.split('/')[-1]
    df['SMILES'] = df.apply(pdbqt2pdb, axis=1)
    df.to_csv(os.path.join(output_dir, f'{family}.{target}.csv'), sep='\t', index=False)
