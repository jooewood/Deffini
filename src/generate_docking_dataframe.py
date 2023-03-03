#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import glob
import pandas as pd
import numpy as np


"""
------------------------------------------------------------------------------
Kernie
------------------------------------------------------------------------------
"""

input_dir = '/y/Aurora/Fernie/Kernie'


active_pdbqt_paths = glob.glob(os.path.join(input_dir, '*', '*', 'cmpd_library', 'active', '*'))
decoy_pdbqt_paths = glob.glob(os.path.join(input_dir, '*', '*', 'cmpd_library', 'decoy', '*'))



pdbqt_paths_before_docking = active_pdbqt_paths + decoy_pdbqt_paths

active_df = pd.DataFrame({
    'cmpd_pdbqt_before_docking': active_pdbqt_paths,
    'label': 1
    })
decoy_df = pd.DataFrame({
    'cmpd_pdbqt_before_docking': decoy_pdbqt_paths,
    'label':0
    })

def get_target_name(x):
    tmp = x.split('/')
    input_dir = '/y/Aurora/Fernie/Kernie'
    target_name = tmp[5]
    conformation = tmp[6]
    cmpd_file = tmp[-1]
    type_ = tmp[-2]
    cmpd_library = 'cmpd_library'
    docking_pdbqt = 'docking_pdbqt'
    return pd.Series([input_dir, target_name, conformation, type_, cmpd_file, cmpd_library, docking_pdbqt])

df = pd.concat([active_df, decoy_df])

df[['input_dir', 'target_name', 'conformation', 'cmpd_type', 'cmpd_file', 'cmpd_library', 'docking_pdbqt']] =  df['cmpd_pdbqt_before_docking'].apply(get_target_name)
del df['cmpd_pdbqt_before_docking']
df['target_path'] = 'target/receptor.pdbqt'
df['ref_ligand'] = 'target/ref_ligand.pdbqt'
del df['target']
df['tmp_dir'] = 'tmp'
df.to_csv('Kernie_docking_guideline.tsv', sep='\t', index=False)
