#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import pandas as pd
from glob import glob

files = glob('/y/Aurora/Fernie/data/ligand_based_data/Kernie/*.csv')

dfs = []
l = []
for file in files:
    df = pd.read_csv(file, sep='\t')
    df = df.query('label == 1')
    l.append(len(df))
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv('/y/Aurora/Fernie/data/ligand_based_data/Kernie_active.csv', sep='\t', index=False)
df = df[['SMILES', 'ID']]
df.to_csv('/y/Aurora/Fernie/data/ligand_based_data/Kernie_active_without_header.csv', sep='\t', header=False, index=False)



files = glob('/y/Aurora/Fernie/data/ligand_based_data/DUD-E/*.csv')

dfs = []
l = []
for file in files:
    df = pd.read_csv(file, sep='\t')
    df = df.query('label == 1')
    l.append(len(df))
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv('/y/Aurora/Fernie/tune/ZINC/DeepCoy/data/DUD-E/DUD-E_active.csv', index=False)
df = df[['SMILES']]
df.to_csv('/y/Aurora/Fernie/tune/ZINC/DeepCoy/data/DUD-E/DUD-E_active.smi', index=False, header=None)
