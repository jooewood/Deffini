#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

import os
import sys
import math
import glob

def rm_space_of_one_variable(x):
    return x.strip()

def extract(data, whole_data): # extract data from 'whole_data' according to 'data'
    extract_data = []
    for i in data:
        for j in whole_data:
            if i in j:
                extract_data.append(j)
    return extract_data

def remain_items_that_are_dir(x):
	def judge_whether_is_dir(path):
		if not os.path.isdir(path):
			return False
		else:
			return True
	return list(filter(judge_whether_is_dir, x))

def get_task_folders(data_root):
    targets = os.listdir(data_root) 
    targets_path = []
    for target in targets: # delete items which are not target folder
        path = '%s/%s'%(data_root, target)
        if os.path.isdir(path):
            targets_path.append(path)
        else:
            targets.remove(target)
    ## get depth by dataset
    dataset = data_root.split('/')[-1]
    if dataset =='kinformation': # set depth of dataset
        depth = 2
    elif dataset == 'DUD-E':
        depth = 1
    ## get task folders by depth
    path = data_root
    for _ in range(depth):
        path = path + '/*'
    task_folders = remain_items_that_are_dir(glob.glob(path)) # remove path that are not direction
    return task_folders

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

def main(dataset):

    working_dir = '/y/Fernie/scratch'
    root_path = '/y/Fernie/kinase_project/data'
    dataset_output_dir = os.path.join(working_dir, dataset)
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)

    data_root = os.path.join(root_path,dataset)
    path = os.path.join(working_dir,dataset)
    out_path = os.path.join(path+ '_clstr_0.5')
    fst_path = os.path.join(path+'.fst')
    total_num = 0
    with open(fst_path, 'r') as f:
        for i in f.readlines():
            if '>' in i:
                total_num += 1
    # dataset = data_root.split('/')[-1]
    fold_1, fold_2, fold_3 = get_fold_of_protein(out_path, total_num)
    all_task_folders = get_task_folders(data_root)
    test_0 = extract(fold_1, all_task_folders)
    test_1 = extract(fold_2, all_task_folders)
    test_2 = extract(fold_3, all_task_folders)
    train_0 = test_1 + test_2
    train_1 = test_0 + test_2
    train_2 = test_0 + test_1


    outdir_1 = os.path.join(working_dir, 'cluster_3_fold')
    outdir_2 = os.path.join(outdir_1, dataset)
    output_folder = os.path.join(outdir_2, 'split_files')

    if not os.path.exists(outdir_1):
        os.makedirs(outdir_1)
    if not os.path.exists(outdir_2):
        os.makedirs(outdir_2)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(3):
        test_file_path = os.path.join(output_folder, 'test_'+str(i)+'.taskfolders')
        train_file_path = os.path.join(output_folder, 'train_'+str(i)+'.taskfolders')
        write_out(test_file_path, locals()['test_%d' % i])
        write_out(train_file_path, locals()['train_%d' % i])


if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")