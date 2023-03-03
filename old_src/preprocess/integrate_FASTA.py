#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================
import os
import sys
import glob
import pickle
import numpy as np

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
    data_root = os.path.join(root_path,dataset)

    path = os.path.join(working_dir,dataset)
    path_out = os.path.join(path+'.fst')

    if dataset=='DUD-E':
        data_root = os.path.join(path, '*')
        task_folders = remain_items_that_are_dir(glob.glob(data_root))
        f = open(path_out, 'w')
        for task_folder in task_folders:
            # task_folder = task_folders[0]
            target = task_folder.split('/')[-1]
            fasta_path = glob.glob('{}/target/*.fst'.format(task_folder))[0]
            f_tmp = open(fasta_path, 'r')
            content = f_tmp.readlines()
            for line in content:
                # line = content[0]
                if '>' in line:
                    print('>%s'%target, file=f)
                else:
                    line = line.strip()
                    f.write(line)
                    
            f.write('\n')
        f.close()
    elif dataset=='kinformation':
        data_root = os.path.join(path, '*', '*')
        task_folders = remain_items_that_are_dir(glob.glob(data_root))    
        targets = []
        pre_target = None
        f = open(path_out, 'w')
        for i, task_folder in enumerate(task_folders):
            # task_folder = task_folders[0]
            target = task_folder.split('/')[-2]
            if target==pre_target:
                continue
            targets.append(target)
            pre_target = target
            fasta_path = glob.glob('{}/target/*.fst'.format(task_folder))[0]
            f_tmp = open(fasta_path, 'r')
            content = f_tmp.readlines()
            for line in content:
                # line = content[0]
                if '>' in line and i==0:
                    print('>%s'%target, file=f)
                elif '>' in line:
                    print('\n>%s'%target, file=f)
                else:
                    line = line.strip()
                    f.write(line)
        f.close()


if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")