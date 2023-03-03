#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

import feature_extractor_no_grid_train as FE
import os
import glob
from multiprocessing import cpu_count
import multiprocessing
import random
from tqdm import tqdm

def diff(a,b):# a must bigger than b
    return list(set(a).difference(set(b)))

def batch_predata(task_folders, output_dir, depth, multi_pose):
    fail = []
    print("start to feature extract...")
    for task_folder in tqdm(task_folders):
        print("Handle the %s" % task_folder)
        if task_folder[-1] == '/':
            task_folder = task_folder[:-1]
        task_folder_taken = task_folder.split('/')
        start = -depth
        tmp_output_dir = output_dir
        for i in range(depth):
            tmp_output_dir=os.path.join(tmp_output_dir,task_folder_taken[start])
            start += 1   
        tmp_output_dir = os.path.join(tmp_output_dir, 'non_grid_feature')
        try: 
            FE.get_predata(task_folder, tmp_output_dir, MULTI_POSE=multi_pose)
            print("Handle the %s succeeded." % task_folder)
        except:
            print("Handle the %s failed." % task_folder)
            fail.append(task_folder)
    return fail

def multi_process_zdx(function_name, cpu_number, *args):
    """
    'function_name' is a function you create.
    """
    split_number = cpu_number
    ## First, we need to split input_data
    surplus = args[0].copy()
    l = len(surplus)
    segment_l = int(l/split_number)
    for i in range(split_number):
        if not i==split_number-1:
            tmp_input = random.sample(surplus, segment_l)
            surplus = diff(surplus, tmp_input)
        else:
            tmp_input = surplus
        p = multiprocessing.Process(target=function_name, args=(tmp_input, 
                                                                args[1], 
                                                                args[2], 
                                                                args[3]))
        p.start()

def main(input_dir, output_dir, depth, multi_pose, cpu_number):
    for i in range(depth):
        input_dir = os.path.join(input_dir, '*')
    task_folders = glob.glob(input_dir)
    multi_process_zdx(batch_predata, cpu_number, task_folders, output_dir, 
        depth, multi_pose)

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()  
    ap.add_argument("-i", "--input_dir", type=str, required=True,
                    help="The location of dataset.")
    ap.add_argument("-o", "--output_dir", type=str, required=True,
                    help="The location which to save the extracted feature.")
    ap.add_argument("-d", "--depth", type=int, default=2,
                    help="The depth of task_folder")
    ap.add_argument("-m", "--multi_pose", action='store_true', default=False,
                    help="Whether extract multi_pose of compounds")

    args = ap.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        depth = args.depth,
        multi_pose = args.multi_pose,
        cpu_number = args.cpu_number
    )
    
    
import feature_extractor_no_grid_train as FE
import os
import glob
from tqdm import tqdm

def diff(a,b):# a must bigger than b
    return list(set(a).difference(set(b)))

def batch_predata(task_folders, output_dir, depth, multi_pose):
    # task_folders = task_folders[:1]
    # task_folders = '/y/Fernie/kernie_mini/A3FQ16/codi'
    # input_dir = '/y/Fernie/kernie_mini'
    # output_dir = '/y/Fernie/output/kernie_output'
    fail = []
    print("start to feature extract...")
    for task_folder in tqdm(task_folders):
        print("Handle the %s" % task_folder)
        if task_folder[-1] == '/':
            task_folder = task_folder[:-1]
        task_folder_taken = task_folder.split('/')
        start = -depth
        tmp_output_dir = output_dir
        for i in range(depth):
            tmp_output_dir = os.path.join(tmp_output_dir, task_folder_taken[start])
            start += 1   
        tmp_output_dir = os.path.join(tmp_output_dir, 'non_grid_feature')
        try: 
            FE.get_predata(task_folder, tmp_output_dir, MULTI_POSE=multi_pose)
            print("Handle the %s succeeded." % task_folder)
        except:
            print("Handle the %s failed." % task_folder)
            fail.append(task_folder)
    return fail


def main(input_dir, output_dir, depth, multi_pose):
    for i in range(depth):
        input_dir = os.path.join(input_dir, '*')
    task_folders = glob.glob(input_dir)
    batch_predata(task_folders, output_dir, depth, multi_pose)

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()  
    ap.add_argument("-i", "--input_dir", type=str, required=True,
                    help="The location of dataset.")
    ap.add_argument("-o", "--output_dir", type=str, required=True,
                    help="The location which to save the extracted feature.")
    ap.add_argument("-d", "--depth", type=int, default=2,
                    help="The depth of task_folder")
    ap.add_argument("-m", "--multi_pose", type=False, action='store_true',
                    help="Whether extract multi_pose of compounds")
    ap.add_argument("-c", "--cpu_number", default=cpu_count()-1, type=int, 
                    help="The number of CPUs you want to use.")
    args = ap.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    main(
        input_dir = args.input_dir,
        output_dir = args.output_dir,
        depth = args.depth,
        multi_pose = args.multi_pose,
        cpu_number = args.cpu_number
    )
