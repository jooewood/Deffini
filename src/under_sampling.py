#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
from utils import build_new_folder

def under_sample_a_target(input_file, output_file):
    with open(input_file, 'rb') as fb:
        one_target_feature = pickle.load(fb)
    active_index = np.where(one_target_feature[6][:, 1] == 1)[0]
    active_len = len(active_index)
    decoy_index = np.where(one_target_feature[6][:, 1] == 0)[0]
    decoy_len = len(decoy_index)
    rate = int(decoy_len/active_len)
    print(f"Before undersample.\n {active_len} actives \n {decoy_len} decoys. \n"
          f"Decoy:Active is {rate}:1")
    
    sampled_decoy_index = np.random.choice(decoy_index, size=active_len)
    sampled_decoy_len = len(sampled_decoy_index)
    rate = sampled_decoy_len/active_len
    # print(f"After undersample.\n {active_len} actives \n {decoy_len} decoys. \n"
    #       f"Decoy:Active is {rate}:1")
    
    sampled_index = np.concatenate((active_index, sampled_decoy_index), axis=0)
    
    sampled_feature = []
    for i in range(7):
        sampled_feature.append(
            np.array([one_target_feature[i][x] for x in sampled_index])
            )
    active_len = np.sum(sampled_feature[6][:, 1] == 1)
    decoy_len = np.sum(sampled_feature[6][:, 1] == 0)
    rate = int(active_len/decoy_len)
    print(f"\nAfter undersample.\n{active_len} actives\n{decoy_len} decoys.\n"
          f"Decoy:Active is {rate}:1\n\n--------------------------------------")
    with open(output_file, 'wb') as fb:
        pickle.dump(sampled_feature, fb, protocol=4)

def under_sample_a_dir(input_dir, output_dir):
    build_new_folder(output_dir)
    files = glob.glob(os.path.join(input_dir, '*.pickle'))
    for file in tqdm(files):
        # file = files[0]
        file_name, suffix = os.path.splitext(os.path.basename(file))
        outputfile = os.path.join(output_dir, f'{file_name}.undersample.pickle')
        under_sample_a_target(file, outputfile)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-ir', '--input_dir', default=None)
    parser.add_argument('-or', '--output_dir', default=None)
    parser.add_argument('-if', '--input_file', default=None)
    parser.add_argument('-of', '--output_file', default=None)
    args = parser.parse_args()
    if args.input_dir is not None:
        under_sample_a_dir(input_dir=args.input_dir,
            output_dir=args.output_dir)
    elif args.input_file is not None:
        under_sample_a_target(args.input_file, args.output_file)
    else:
        print('Please check your input path!')