#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pickle
import numpy as np

def main(input_file):
    with open(input_file, 'rb') as fb:
        one_target_feature = pickle.load(fb)
    active_index = np.where(one_target_feature[6][:, 1] == 1)[0]
    active_len = len(active_index)
    decoy_index = np.where(one_target_feature[6][:, 1] == 0)[0]
    decoy_len = len(decoy_index)
    rate = int(decoy_len/active_len)
    print(f"\n\t{active_len} actives \n\t{decoy_len} decoys."
          f"\n\tDecoy:Active = {rate}:1\n")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input_file', default=None)
    args = parser.parse_args()
    main(args.input_file)