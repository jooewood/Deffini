#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from glob import glob
import pandas as pd
from tqdm import tqdm
import os
from argparse import ArgumentParser

def merge_dfs(subfiles):
    dfs = []
    for file in tqdm(subfiles):
        dfs.append(pd.read_csv(file, sep='\t'))
    merge_dfs = pd.concat(dfs)
    return merge_dfs

def main(input_dir, cluster_dir, output_dir):
    cluster_files = glob(os.path.join(cluster_dir,'*.targets'))
    # screen out test_0.csv...
    test_cluster_files = [x for x in cluster_files if 'test' in x] 
    print('start to merge multiple targets data...')
    for test_file in test_cluster_files:
        subfiles = []
        print(f'start to process {test_file}')
        with open(test_file, 'r') as f:
            sub_targets = f.read().splitlines() # remove \n and return list
        for target in sub_targets:
            subfiles.append(os.path.join(input_dir, f'{target}.csv'))
        outputfile_name = os.path.splitext(os.path.basename(test_file))[0]
        print(f'start to merge files according to {test_file}')
        merged_df = merge_dfs(subfiles)
        outfile = os.path.join(output_dir, f'{outputfile_name}.csv')
        print(f'start to save out merged file according to {test_file}')
        merged_df.to_csv(outfile, index=False, sep='\t')
    print('Finished.')
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Path to targets CSV file')
    parser.add_argument('-c', '--cluster_dir', type=str, required=True,
                        help='Path to test CSV file including targets name')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Path to the result of merge test targets CSV file')
    args = parser.parse_args()
    main(input_dir=args.input_dir, 
         cluster_dir=args.cluster_dir, 
         output_dir=args.output_dir)