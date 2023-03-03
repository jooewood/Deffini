#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import glob
import pandas as pd
from tqdm import tqdm
from single_target_feature_extractor import feature_extractor
from utils import parallelize_dataframe

def extract_one_target_feature(task_folder, output_dir, depth, multi_pose, 
                               labeled):
    docking_pdbqts = glob.glob(os.path.join(task_folder, 
                                            'docking_pdbqt/*/*.pdbqt'))
    target_pdbqt = os.path.join(task_folder, 'target/receptor.pdbqt')
    folder = task_folder.split('/')
    name = []
    if depth>1:
        for i in range(depth):
            name.append(folder.pop())
            name.reverse()
        file_name = '.'.join(name)
    else:
        file_name = folder.pop()
    if multi_pose:
        output_file = os.path.join(output_dir, f'{file_name}.mul.pickle')
    else:
        output_file = os.path.join(output_dir, f'{file_name}.sin.pickle')

    return feature_extractor(
        protein_file=target_pdbqt,
        ligand_files=docking_pdbqts, 
        multi_pose=multi_pose, 
        labeled=labeled, 
        pickle_filename=output_file)


def main(input_dir, output_dir, depth, multi_pose=False, labeled=True, 
         sequential=True):
    input_dir_tmp = input_dir
    for i in range(depth):
        input_dir_tmp = os.path.join(input_dir_tmp, '*')
    task_folders = glob.glob(input_dir_tmp)
    # a_docking_pdbqts = glob.glob(os.path.join(input_dir_tmp, 
    #                                         'docking_pdbqt/active/*.pdbqt'))
    # d_docking_pdbqts = glob.glob(os.path.join(input_dir_tmp, 
    #                                         'docking_pdbqt/decoy/*.pdbqt'))
    # print(len(a_docking_pdbqts))
    # print(len(d_docking_pdbqts))
    
    if sequential:
        for task_folder in tqdm(task_folders):
            extract_one_target_feature(task_folder, output_dir, depth, 
                                       multi_pose, labeled)
    else:
        df = pd.DataFrame({
            'task_folder': task_folders,
            'outout_dir': output_dir,
            'depth': depth,
            'multi_pose': multi_pose,
            'labeled': labeled
            })
        
        def df_extract_feature_each_line(line):
            return extract_one_target_feature(task_folder = line['task_folder'],
                                              outout_dir = line['outout_dir'],
                                              depth = line['depth'],
                                              multi_pose = line['multi_pose'],
                                              labeled = line['labeled']
                                              )
            
        
        def parall_extract_feature(df):
            df['extract_feature'] = df.apply(df_extract_feature_each_line, axis=1)
            return df
        
        df = parallelize_dataframe(df, parall_extract_feature)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir')
    parser.add_argument('-s', '--sequential', action='store_false', default=True)
    parser.add_argument('-d', '--depth', type=int, default=2)
    parser.add_argument('-m', '--multi_pose', action='store_true', default=False,
        help="Whether to extract multi binding pose feature.")
    parser.add_argument('-labeled', action='store_false', default=True,
        help="Whether to add dataset label in the feature file.")
    args = parser.parse_args()
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         depth=args.depth,
         multi_pose=args.multi_pose,
         labeled=args.labeled,
         sequential=args.sequential
         )
    
