#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a scipt about prepare data for increaseing targets experiments

@author: zdx
"""
import os
from glob import glob
from package_data import package_targets_feature
import pandas as pd
import random
from functools import partial
from utils import try_copy
from multiprocessing import Pool, Process, cpu_count

def get_sub_pickles(target, feature_dir):
    sub_pickles = glob(os.path.join(feature_dir, f'{target}.*'))
    return sub_pickles

def prepare_increase_targets_data(p, targets, feature_dir, output_dir, 
                                  consisent_quantity=False):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(os.path.join(output_dir, 'targets.order'), 'w') as f:
        for x in targets:
            print(x, file=f)
    
    l = len(targets)
        
    for i in range(l):
        print(i)
        if i==l:
            break
        else:
            j = i+1
        sub_targets = targets[0:j]
        
        
        sub_pickles = []
        for target in sub_targets:
            sub_pickles += get_sub_pickles(target, feature_dir)
        if len(sub_pickles)==1:
            if p is None:
                try_copy(sub_pickles[0], 
                         os.path.join(output_dir, '%s.pickle' % str(j)))
            else:
                p.apply_async(try_copy, args=(sub_pickles[0], 
                        os.path.join(output_dir, '%s.pickle' % str(j)), )
                    )
        else:
            if p is None:
                package_targets_feature(sub_pickles, 
                    os.path.join(output_dir, '%s.pickle' % str(j)))
            else:
                p.apply_async(package_targets_feature, args=(sub_pickles, 
                    os.path.join(output_dir, '%s.pickle' % str(j)),))

def repeat_prepare_increase_targets_data(targets, feature_dir, output_dir, 
    repeat_time=5, consisent_quantity=False, multi_process=False):
    if multi_process:
        p = Pool(cpu_count()-1)
        print('Waiting for all subprocesses done...')
        for i in range(repeat_time):
            random.shuffle(targets)
            tmp_output_dir = os.path.join(output_dir, str(i))
            prepare_increase_targets_data(p, targets, feature_dir, tmp_output_dir, 
                                      consisent_quantity=consisent_quantity)
        p.close()
        p.join()
        print('All subprocesses done.')
    else:
        p =None
        for i in range(repeat_time):
            random.shuffle(targets)
            tmp_output_dir = os.path.join(output_dir, str(i))
            prepare_increase_targets_data(p, targets, feature_dir, tmp_output_dir, 
                                      consisent_quantity=consisent_quantity)


"""
Kernie undersample
"""

data_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie'
targets = os.listdir(data_dir)
# with open('/y/Aurora/Fernie/data/Kernie.UniprotID', 'w') as f:
#     for x in targets:
#         print(x, file=f)

# feature_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose_under_sampling'
# output_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/increase_targets/undersample'

# repeat_prepare_increase_targets_data(targets, feature_dir, output_dir)

"""
Kernie raw
"""
print('Kernie raw')
feature_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/single_pose'
output_dir = '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature/increase_targets/raw'
repeat_prepare_increase_targets_data(targets, feature_dir, output_dir)

"""
DUD-E undersample
"""
# print('DUD-E undersample')
# csv_file = '/y/Aurora/Fernie/data/DUD-E_info.csv'

# df = pd.read_csv(csv_file)
# feature_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/single_pose_under_sampling'
# output_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/increase_targets/'

# familys = list(set(df['family']))

# for family in familys:
#     # family = familys[0]
#     tmp_output_dir = os.path.join(output_dir, family, 'undersample)
#     sub_targets = df.query('family=="{}"'.format(family))['Target Name']
#     sub_targets = [x.lower() for x in sub_targets]
#     repeat_prepare_increase_targets_data(sub_targets, feature_dir, tmp_output_dir)

"""
DUD-E raw
"""

csv_file = '/y/Aurora/Fernie/data/DUD-E_info.csv'
print('DUD-E raw')
df = pd.read_csv(csv_file)
feature_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/single_pose'
output_dir = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/increase_targets/'

familys = list(set(df['family']))

for family in familys:
    # family = familys[0]
    tmp_output_dir = os.path.join(output_dir, family, 'raw')
    sub_targets = df.query('family=="{}"'.format(family))['Target Name']
    sub_targets = [x.lower() for x in sub_targets]
    repeat_prepare_increase_targets_data(sub_targets, feature_dir, tmp_output_dir)