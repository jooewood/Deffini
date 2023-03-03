#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from glob import glob
import pandas as pd
from package_data import package_targets_feature
from tqdm import tqdm
from utils import try_copy



# ------------
# Divide DUD-E different family undersample
# ------------
def package_according_to_a_df(sub_info, output_dir, undersample_feature_dir,
                              dataset_name, family, undersample):
    sub_targets = [x.lower() for x in sub_info['Target Name']]
    if undersample:
        print('Undersample.')
        undersample_feature_pickles = [os.path.join(undersample_feature_dir,\
            f'{x}.sin.undersample.pickle') for x in sub_targets]
    else:
        undersample_feature_pickles = [os.path.join(undersample_feature_dir,\
            f'{x}.sin.pickle') for x in sub_targets]
    if family is None:
        if undersample:
            print('Undersample.')
            sub_output_file = os.path.join(output_dir, 
                                           f'{dataset_name}.single_pose.undersample.pickle')
        else:
            sub_output_file = os.path.join(output_dir, 
                                           f'{dataset_name}.single_pose.pickle')
    else:
        if undersample:
            print('Undersample.')
            sub_output_file = os.path.join(output_dir, 
                f'{dataset_name}.{family}.single_pose.undersample.pickle')
        else:
            sub_output_file = os.path.join(output_dir, 
                f'{dataset_name}.{family}.single_pose.pickle')
    package_targets_feature(undersample_feature_pickles, sub_output_file)

def handle_one_dataset(info_path, feature_dir, output_dir,
                       dataset_name, undersample=True):
    info = pd.read_csv(info_path)
    package_according_to_a_df(info, output_dir, feature_dir,
                                  dataset_name, None, undersample=undersample)
    familys = set(info['family'])
    for family in tqdm(familys):
        # family = "Kinase"
        sub_info = info.query("family=='%s'" % family)
        package_according_to_a_df(sub_info, output_dir, feature_dir,
                                  dataset_name, family, undersample=undersample)
        
undersample_feature_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/single_pose_under_sampling'
info_path = '/y/Aurora/Fernie/data/DUD-E_info.csv'
output_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature'
dataset_name = 'DUD-E'

# handle_one_dataset(info_path, undersample_feature_dir, output_dir,
#                        dataset_name)
# ------------
# Divide DUD-E different family without undersample
# ------------

feature_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/single_pose'
handle_one_dataset(info_path, feature_dir, output_dir,
                       dataset_name, undersample=False)
# -------------
# whole Kernie undersample
# -------------
undersample_feature_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose_under_sampling'
output_file = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/Kernie.single_pose.undersample.pickle'
pickle_paths = glob(os.path.join(undersample_feature_dir, '*.pickle'))

package_targets_feature(pickle_paths, output_file)
# -------------
# Kernie-MUV undersample
# -------------
sub_pickle_paths = []
for x in pickle_paths:
    if 'P17612' in x or 'P61925' in x or 'O75116' in x or 'Q03137' in x or\
        'Q05397' in x :
        continue
    else:
        sub_pickle_paths.append(x)
output_file = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/Kernie-MUV.single_pose.undersample.pickle'
package_targets_feature(sub_pickle_paths, output_file)

# -------------
# Kernie - MUV 
# -------------
feature_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose'
output_file = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/Kernie-MUV.single_pose.pickle'
pickle_paths = glob(os.path.join(feature_dir, '*.pickle'))

sub_pickle_paths = []
for x in pickle_paths:
    if 'P17612' in x or 'P61925' in x or 'O75116' in x or 'Q03137' in x or\
        'Q05397' in x :
        continue
    else:
        sub_pickle_paths.append(x)
output_file = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/Kernie-MUV.single_pose.pickle'
package_targets_feature(sub_pickle_paths, output_file)

# ------------
# Divide MUV into MUV_feature single pose and multi_pose

pickle_paths = glob('/y/Aurora/Fernie/data/structure_based_data/MUV/*/*/tmp/*.pickle')
output_dir = '/y/Aurora/Fernie/data/structure_based_data/MUV_feature'

for path in tqdm(pickle_paths):
    # path = pickle_paths[0]
    items_ = path.split('/')
    family = items_[7]
    target_name = items_[8]
    if 'multi' in path:
        target = os.path.join(output_dir, 'multi_pose', f'{target_name}.{family}.mul.pickle')
        try_copy(path, target)
    else:
        target = os.path.join(output_dir, 'single_pose', f'{target_name}.{family}.sin.pickle')
        try_copy(path, target)
        
        
# -----------------
# Package different MUV family
pickle_paths = glob('/y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose/*.pickle')
output_dir = '/y/Aurora/Fernie/data/structure_based_data/MUV_feature'
MUV_info = pd.read_excel('/y/Aurora/Fernie/data/MUV_info.xlsx')
familys = list(set(MUV_info['family']))
familys = [x.strip() for x in familys]
for family in familys:
    print(family)
    sub_files = []
    for pickle_path in pickle_paths:
        if family in pickle_path:
            sub_files.append(pickle_path)
    print(len(sub_files))
    output_file = os.path.join(output_dir, f'MUV.{family}.pickle')
    if len(sub_files)==1:
        try_copy(sub_files[0], output_file)
    elif len(sub_files)>1:
        package_targets_feature(sub_files, output_file)
    
pickle_paths = glob('/y/Aurora/Fernie/data/structure_based_data/MUV_feature/single_pose_undersampling/*.pickle')
output_dir = '/y/Aurora/Fernie/data/structure_based_data/MUV_feature'
MUV_info = pd.read_excel('/y/Aurora/Fernie/data/MUV_info.xlsx')
familys = list(set(MUV_info['family']))
familys = [x.strip() for x in familys]
for family in familys:
    print(family)
    sub_files = []
    for pickle_path in pickle_paths:
        if family in pickle_path:
            sub_files.append(pickle_path)
    print(len(sub_files))
    output_file = os.path.join(output_dir, f'MUV.{family}.undersample.pickle')
    if len(sub_files)==1:
        try_copy(sub_files[0], output_file)
    elif len(sub_files)>1:
        package_targets_feature(sub_files, output_file)