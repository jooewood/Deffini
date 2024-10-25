#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

import os
import glob

def remain_items_that_are_dir(x):
	def judge_whether_is_dir(path):
		if not os.path.isdir(path):
			return False
		else:
			return True
	return list(filter(judge_whether_is_dir, x))

def main(dataset):
    # dataset = "kinformation"
    root_path = os.path.dirname(os.path.dirname(os.getcwd()))
    data_root = root_path + '/data/%s'%dataset
    if dataset=="DUD-E":
        path = data_root + '/*'
    elif dataset=="kinformation":
        path = data_root + '/*/*'
    task_folders = remain_items_that_are_dir(glob.glob(path))
    failed = []
    for task_folder in task_folders:
        active_path = task_folder + '/scoring_smina/active.smina.tsv'
        decoy_path = task_folder + '/scoring_smina/decoy.smina.tsv'
        if not os.path.exists(active_path) or not os.path.getsize(active_path):
            failed.append(task_folder)
        if not os.path.exists(decoy_path) or not os.path.getsize(decoy_path):
            failed.append(task_folder)

    
        
if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")