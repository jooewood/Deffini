#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from multiprocessing import Pool
from multiprocessing import Process
import _thread
import glob
import os
from package_data import package_targets_feature
from utils import try_copy
from tqdm import tqdm
from utils import build_new_folder

def get_file(feature_files, cluster_file):
    sub_files = []
    with open(cluster_file, 'r') as f:
        subtargets = f.read().splitlines()
        for target in subtargets:
            name = target + '.'
            for file in feature_files:
                if name in file:
                    sub_files.append(file)
    return sub_files


def copy_test_files(raw_files, output_dir):
    build_new_folder(output_dir)
    print(f'start to copy into {output_dir}')
    for file in tqdm(raw_files):
        file_name = os.path.basename(file)
        copy_target_file = os.path.join(output_dir, file_name)
        try_copy(file, copy_target_file)

def merge_data_according_to_a_targets_list(feature_files, cluster_file, 
                                           output_dir, copy, undersample=True):
    file_name = os.path.splitext(os.path.basename(cluster_file))[0]
    sub_files = get_file(feature_files, cluster_file)
    if undersample:
        output_file = os.path.join(output_dir, f'{file_name}.pickle')
    else:
        output_file = os.path.join(output_dir, f'{file_name}.raw.pickle')
    print(f'start to package into {output_file}')

    if not copy:
        p = Process(target=package_targets_feature, args=(sub_files, output_file, ))
        p.start()

    sub_output_dir = os.path.join(output_dir, file_name)
    if copy:
        p = Process(target=copy_test_files, args=(sub_files, sub_output_dir, ))
        p.start()

def get_sub_targets(feature_files, cluster_files, output_dir, copy=False, 
                    undersample=True):
    for cluster_file in cluster_files:
        merge_data_according_to_a_targets_list(feature_files, 
                                               cluster_file, output_dir, copy, 
                                               undersample)

def generate_cv_data(raw_feature_dir, under_sample_dir, cluster_file_dir, 
                     output_dir, copy=False, undersample=True):
    # get cluster file path, which includes targets' names
    cluster_files = glob.glob(os.path.join(cluster_file_dir, '*.targets'))

    # Divide into train and test cluster files
    train_cluster_files = [x for x in cluster_files if 'train' in x]
    test_cluster_files = [x for x in cluster_files if 'test' in x]

    undersample_files = glob.glob(os.path.join(under_sample_dir, '*.pickle'))
    
    raw_files = glob.glob(os.path.join(raw_feature_dir, '*.pickle'))

    # Generate train package pickle data
    if undersample:
        get_sub_targets(undersample_files, train_cluster_files, output_dir)
    else:
        print('Without undersample')
        get_sub_targets(raw_files, train_cluster_files, output_dir, 
                        undersample=False)
    # Generate test package pickle data
    # get_sub_targets(undersample_files, test_cluster_files, output_dir)

    # Copy test file into destination
    # get_sub_targets(raw_files, test_cluster_files, output_dir, copy=True)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--undersample', default=1, type=int, choices=[0,1])
    parser.add_argument('-r', '--raw_feature_dir')
    parser.add_argument('-u', '--under_sample_dir')
    parser.add_argument('-c', '--cluster_file_dir')
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()
    generate_cv_data(raw_feature_dir=args.raw_feature_dir, 
                     under_sample_dir=args.under_sample_dir, 
                     cluster_file_dir=args.cluster_file_dir, 
                     output_dir=args.output_dir,
                     undersample=args.undersample
                     )
    
"""
Generate cross validation data
------------------------------

dataset=DUD-E
./generate_cv_data.py -r /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/${dataset}_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/${dataset}_3_fold
dataset=Kernie
./generate_cv_data.py -r /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/${dataset}_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/${dataset}_3_fold

dataset=DUD-E &&\
./generate_cv_data.py --undersample 0 -r /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/${dataset}_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/${dataset}_3_fold &&\
dataset=Kernie &&\
./generate_cv_data.py --undersample 0 -r /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose -u /y/Aurora/Fernie/data/structure_based_data/${dataset}_feature/single_pose_under_sampling -c /y/Aurora/Fernie/data/${dataset}_clur_3_fold -o /y/Aurora/Fernie/data/structure_based_data/${dataset}_3_fold


"""