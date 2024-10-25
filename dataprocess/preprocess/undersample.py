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
import random

def remain_items_that_are_dir(x):
	def judge_whether_is_dir(path):
		if not os.path.isdir(path):
			return False
		else:
			return True
	return list(filter(judge_whether_is_dir, x))

def main(dataset):
    # data_path = '/y/Fernie/kinase_project'
    # data_root = os.path.join(data_path, 'data', dataset)
    working_dir = '/y/Fernie/scratch'
    out_dir = os.path.join(working_dir, dataset)
    ## get dataset and its depth
    if dataset=='DUD-E':
        depth = 1
    elif dataset=='kinformation':
        depth = 2
    ## get task folders by depth
    # path = data_root
    # for _ in range(depth):
    #     path = path + '/*'
    for _ in range(depth):
        out_dir = out_dir + '/*'
    # task_folders = remain_items_that_are_dir(glob.glob(path))
    # remove path that are not direction
    out_folders = remain_items_that_are_dir(glob.glob(out_dir))
    failed = []
    for out_folder in out_folders:
        # task_folder = task_folders[0]
        data_name = out_folder + '/tmp/train.no_grid.pickle'
        active_index = []
        decoy_index = []
        try:
        	with open(data_name, 'rb') as fb:
        		one_target_feature = pickle.load(fb)
        	for i in range(0, len(one_target_feature[6])):
        		item = one_target_feature[6][i]
        		# item = one_target_feature[6][0]
        		if (item==[0,1]).all():
        			active_index.append(i)
        		elif (item==[1,0]).all():
        			decoy_index.append(i)
        	sample_n = len(active_index)
        	new_name = one_target_feature[0][:2*sample_n]
        	new_atm = one_target_feature[1][:2*sample_n]
        	new_chrg = one_target_feature[2][:2*sample_n]
        	new_dist = one_target_feature[3][:2*sample_n]
        	new_amino = one_target_feature[4][:2*sample_n]
        	new_mask_mat = one_target_feature[5][:2*sample_n]
        	new_y = one_target_feature[6][:2*sample_n]
        	path_out = '{}/tmp/feature_undersample.pickle'.format(out_folder)
        	with open(path_out, 'wb') as fb:
        		pickle.dump([new_name, new_atm, new_chrg, new_dist, new_amino, \
        		new_mask_mat, new_y], fb, protocol=4)
        except:
        	failed.append(data_name)
    l_fail = len(failed)
    print("\nUndersampling finished. {} failed.\n".format(l_fail))
    if l_fail != 0:
        print(failed)


if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")