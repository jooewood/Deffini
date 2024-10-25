#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

import os
import sys
import glob
from subprocess import call

def remain_items_that_are_dir(x):
	def judge_whether_is_dir(path):
		if not os.path.isdir(path):
			return False
		else:
			return True
	return list(filter(judge_whether_is_dir, x))

def main(dataset):
    working_dir = '/y/Fernie/scratch'
    root_path = '/y/Fernie/kinase_project/data'
    dataset_output_dir = os.path.join(working_dir, dataset)
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)
    data_root = os.path.join(root_path,dataset)
    if dataset=='DUD-E':
        path = os.path.join(data_root, '*')
    elif dataset=='kinformation':
        path = os.path.join(data_root, '*', '*')
    else:
        path = os.path.join(data_root, '*')
    
    task_folders = remain_items_that_are_dir(glob.glob(path))

    for task_folder in task_folders:
        # task_folder = task_folders[0]
        if dataset=='kinformation':
            kdataset_output_dir = os.path.join(dataset_output_dir, 
                task_folder.split('/')[-2])
            target_output_dir = os.path.join(kdataset_output_dir, 
                task_folder.split('/')[-1])
            if not os.path.exists(kdataset_output_dir):
                os.makedirs(kdataset_output_dir)
            if not os.path.exists(target_output_dir):
                os.makedirs(target_output_dir)

        else:
            target_output_dir = os.path.join(dataset_output_dir, 
                task_folder.split('/')[-1])
            if not os.path.exists(target_output_dir):
                os.makedirs(target_output_dir)
        
        fasta_output_dir = os.path.join(target_output_dir, 
            'target')
        if not os.path.exists(fasta_output_dir):
            os.makedirs(fasta_output_dir)

        # print("dataset mkdir done")
        pdb_path = os.path.join(task_folder,'target','receptor.pdb')
        os.path.exists(pdb_path)
        fasta_path = os.path.join(fasta_output_dir,'receptor.fst')
        # print(os.getcwd())

        os.system("python3 pdb_tofasta.py {} > {}".format(pdb_path, fasta_path))

if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")