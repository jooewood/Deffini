#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================


import os
import sys
import datetime
import train
import numpy as np
import pandas as pd
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

def main(dataset):
    project_path = '/y/Fernie/kinase_project'
    working_dir = '/y/Fernie/scratch'
    data_path = os.path.join(project_path, 'cluster_3_fold', dataset, 'data')
    goal_path = os.path.join(working_dir, 'final_score', dataset)
    models_path = os.path.join(goal_path, 'models')

    if not os.path.exists(goal_path):
        os.makedirs(goal_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    #number_of_auroc_over_ninety_final = 0
    for i in range(3):
        # i = 0
        path_to_save_model = os.path.join(models_path, \
            '{}{}.h5'.format(dataset, str(i)))
        train_file = os.path.join(data_path, 'train_{}.pickle'.format( str(i)))
        if not os.path.exists(train_file):
            print("%s not exists." % train_file)
            return
        test_dir = os.path.join(data_path, 'test_{}'.format(str(i)))
        if not os.path.exists(test_dir):
            print("%s not exists." % train_file)
            return
        total_time, failed, ave_auroc, ave_auprc, number_of_auroc_over_ninety \
            = train.train_main(path_to_save_model, train_file, test_dir)

    
if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1";
    main("DUD-E")
    main("kinformation")