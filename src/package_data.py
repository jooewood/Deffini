#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import glob
import numpy as np
import pickle
from tqdm import tqdm


def extract(data, whole_data): # extract data from 'whole_data' according to 'data'
    extract_data = []
    for i in data:
        for j in whole_data:
            if i in j:
                extract_data.append(j)
    return extract_data


def package_targets_feature(targets_feature_files, output_file):
    name_list     = []
    atm_list      = []
    chrg_list     = []
    dist_list     = []
    amino_list    = []
    mask_mat_list = []
    y_list        = []
    for path in tqdm(targets_feature_files):
        with open(path, 'rb') as fb:
            one_target_feature = pickle.load(fb)
        name_list.append(one_target_feature[0])
        atm_list.append(one_target_feature[1])
        chrg_list.append(one_target_feature[2])
        dist_list.append(one_target_feature[3])
        amino_list.append(one_target_feature[4])
        mask_mat_list.append(one_target_feature[5])
        y_list.append(one_target_feature[6])
        
    big_name  = np.concatenate(name_list, axis=0)
    big_atm   = np.concatenate(atm_list, axis=0)
    big_chrg  = np.concatenate(chrg_list, axis=0)
    big_dist  = np.concatenate(dist_list, axis=0)
    big_amino = np.concatenate(amino_list, axis=0)
    big_mask  = np.concatenate(mask_mat_list, axis=0)
    big_y     = np.concatenate(y_list, axis=0)

    packaged_feature =  [big_name, big_atm ,big_chrg, big_dist, big_amino, 
                         big_mask, big_y]
    with open(output_file, 'wb') as fb:
        pickle.dump(packaged_feature, fb, protocol=4)

def main(input_dir, output_file, sub_folder=None, depth=0):
    if sub_folder is None:
        targets_feature_files = glob.glob(os.path.join(input_dir, '*.pickle'))
        package_targets_feature(targets_feature_files, output_file)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', default=None)
    parser.add_argument('-o', '--output_file', help="Packaged feature file.")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_file=args.output_file)