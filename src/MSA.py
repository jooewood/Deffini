#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

from utils import read_fasta, seq2fasta, read_df, get_family, MultiProteinAliganments


file = '/y/Aurora/Fernie/data/DUD-E_MUV.fst'
out_dir = '/y/Aurora/Fernie/Report'
filename = 'DUD-E_MUV'
identities = MultiProteinAliganments(file, filename, out_dir,
                                     figsize=(40, 40))

# info = read_df('/y/Aurora/Fernie/data/DUD-E_info.xlsx')
# names, DUDE_seqs = read_fasta('/y/Aurora/Fernie/data/DUD-E_new.fst')

# name_col = 'Target_Name'
# family_col = 'family'

# familys = get_family(names, info, name_col, family_col)

# DUDE_names = ['_'.join(['DUD-E', name, family]) for name, family in zip(names, familys)]

# names, Kernie_seqs = read_fasta('/y/Aurora/Fernie/data/Kernie_old.fst')
# familys = ['kinase'] * len(names)
# Kernie_names =  ['_'.join(['Kernie', name, family]) for name, family in zip(names, familys)]


# info = read_df('/y/Aurora/Fernie/data/MUV_info.xlsx')
# names, MUV_seqs = read_fasta('/y/Aurora/Fernie/data/MUV.fst')
# name_col = 'confirm_assay_AID'
# family_col = 'family'

# familys = get_family(names, info, name_col, family_col)

# MUV_names =['_'.join(['MUV', name, family]) for name, family in zip(names, familys)]


# names = DUDE_names + MUV_names
# seqs = DUDE_seqs + MUV_seqs
# out = '/y/Aurora/Fernie/data/DUD-E_MUV.fst'
# seq2fasta(seqs, names, out)