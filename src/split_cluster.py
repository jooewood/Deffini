#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import math

# test_file = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/test_0.targets'
# train_file = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/train_0.targets'

# with open(test_file, 'r') as f:
#     test_targets = f.readlines()

# with open(test_file, 'r') as f:
#     train_targets = f.readlines()


def rm_space_of_one_variable(x):
    return x.strip()


def remain_items_that_are_dir(x):
    def judge_whether_is_dir(path):
        if not os.path.isdir(path):
            return False
        else:
            return True
    return list(filter(judge_whether_is_dir, x))


def get_fold_of_protein(path, total_num):
    batchsize = math.ceil(total_num / 3)
    with open(path, 'r') as f:
        content = f.readlines()  
    content.append('>Cluster')
    fold_1 = []
    fold_2 = []
    fold_3 = []
    count = 0
    for i, line in enumerate(content):
        # line = content[1]
        if ">Cluster" in line and count==0:
            tmp_list = []
            count = 1
            continue
        if ">Cluster" in line and count!=0:
            l_tmp_list = len(tmp_list)
            if (len(fold_1) + l_tmp_list) <= batchsize:
                fold_1 += tmp_list
            elif (len(fold_2) + l_tmp_list) <= batchsize:
                fold_2 += tmp_list
            elif (len(fold_3) + l_tmp_list) <= batchsize:
                fold_3 += tmp_list
            tmp_list = []
            continue
        else:
            tmp_protein = rm_space_of_one_variable(line).split('>')[1]
            tmp_protein = tmp_protein.split('...')[0]
            tmp_list.append(tmp_protein)
    fold_1 = list(set(fold_1))
    fold_2 = list(set(fold_2))
    fold_3 = list(set(fold_3))
    return fold_1, fold_2, fold_3

def write_out(path, var):
    with open(path, 'w') as f:
        for item in var:
            print(item, file=f)

cluster_file = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_clstr_0.5'
fst_path = '/y/Aurora/Fernie/data/structure_based_data/DUD-E.fst'
total_num = 0
with open(fst_path, 'r') as f:
    for i in f.readlines():
        if '>' in i:
            total_num += 1
# dataset = data_root.split('/')[-1]
fold_1, fold_2, fold_3 = get_fold_of_protein(cluster_file, total_num)

test_1 = fold_1
train_1 = fold_2 + fold_3
test_2 = fold_2
train_2 = fold_1 + fold_3
test_3 = fold_3
train_3 = fold_1 + fold_2

write_out('/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/test_0.targets', test_1)
write_out('/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/test_1.targets', test_2)
write_out('/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/test_2.targets', test_3)
write_out('/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/train_0.targets', train_1)
write_out('/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/train_1.targets', train_2)
write_out('/y/Aurora/Fernie/data/structure_based_data/DUD-E_clur_3_fold/train_2.targets', train_3)