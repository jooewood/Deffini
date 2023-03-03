#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from glob import glob
import pandas as pd

"""
--------------------------------------------------------------
MUV
--------------------------------------------------------------
"""

input_dir = '/y/Aurora/Fernie/data/ligand_based_data/MUV'

files = glob(os.path.join(input_dir, '*.csv'))

MUV_info = pd.read_excel('/y/Aurora/Fernie/data/MUV_info.xlsx')

MUV_info = MUV_info[['confirm. assay (AID)', 'seq']]
MUV_info.set_index(['confirm. assay (AID)'], inplace=True)
MUV_dict = MUV_info.to_dict()['seq']



for file in files:
    # file = files[0]
    df = pd.read_csv(file, sep='\t')
    target = os.path.basename(file).split('.')[1]
    df['protein'] = MUV_dict[int(target)]
    df.to_csv(file, sep='\t', index=False)
    
    
"""
--------------------------------------------------------------
DUD-E
--------------------------------------------------------------
"""

seq = pd.read_csv('/y/Aurora/Fernie/data/DUD-E.target.sequence.csv')
seq.columns = ['target', 'protein']

seq.set_index(['target'], inplace=True)
seq_dict = seq.to_dict()['protein']

input_dir = '/y/Aurora/Fernie/data/ligand_based_data/DUD-E'

files = glob(os.path.join(input_dir, '*.csv'))

for file in files:
    # file = files[0]
    df = pd.read_csv(file, sep='\t')
    target = os.path.splitext(os.path.basename(file))[0]
    df['protein'] = seq_dict[target]
    df.to_csv(file, sep='\t', index=False)


"""
--------------------------------------------------------------
Kernie
--------------------------------------------------------------
"""

seq = pd.read_csv('/y/Aurora/Fernie/data/Kernie.target.sequence.csv')
seq.columns = ['target', 'protein']

seq.set_index(['target'], inplace=True)
seq_dict = seq.to_dict()['protein']

input_dir = '/y/Aurora/Fernie/data/ligand_based_data/Kernie'

files = glob(os.path.join(input_dir, '*.csv'))

for file in files:
    # file = files[0]
    df = pd.read_csv(file, sep='\t')
    target = os.path.basename(file).split('.')[0]
    df['protein'] = seq_dict[target]
    df.to_csv(file, sep='\t', index=False)