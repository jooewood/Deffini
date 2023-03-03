#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from tqdm import tqdm
from os.path import join, exists, basename, dirname
import torch
from dataset import FernieDataset
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from torch.utils.data import Subset

# input_file = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.single_pose.undersample.pickle'
def kfold_dataset(input_file, n_splits=10, random_state=42):
    file_name = basename(input_file).split('.pickle')[0]
    output_dir = join(dirname(input_file), file_name)
    if not exists(output_dir):
        os.makedirs(output_dir)
    org_dataset = FernieDataset(input_file)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    with tqdm(total=n_splits) as pbar:
        for fold, (train_ids, test_ids) in enumerate(kfold.split(org_dataset)):
            test_dataset = Subset(org_dataset, test_ids)
            file = join(output_dir, f'test_{fold}.pt')
            torch.save(test_dataset, file)
            pbar.update(1)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('-i', '--input_file', required=True)
    args = ap.parse_args()
    kfold_dataset(input_file=args.input_file)