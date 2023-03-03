#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import sys
import datetime
import train
import numpy as np
import pandas as pd
"""
root_path = "/home/tensorflow/kinase_project/3_fold_cross_validation/DUD-E"
hyper_parameter = "batchsize"
value_range = [16, 32, 64, 128, 256, 512, 1024]
fold_number=3

hyper_parameter = "iteration"
value_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
fold_number=3

hyper_parameter = "activation"
value_range = []
"""
def main(root_path, hyper_parameter, value, fold_number=3):
    BATCH_SIZE = 1024
    iteration = 11
    dataset = root_path.split('/')[-1]
    data_folder = '{}/data'.format(root_path) # root_path/data
    parameter_folder = '{}/diff_{}'.format(root_path, hyper_parameter)# root_path/diff_{}
    AUROC = []
    AUPRC = []
    over_90 = []
    Time = []
    for value in value_range:
        try:
            # value = value_range[0]
            if hyper_parameter=='batchsize':
                BATCH_SIZE = int(value)
            elif hyper_parameter=='iteration':
                iteration = int(value)
            model_folder = '{}/{}/models'.format(parameter_folder, value)# root_path/diff_{}/value/models
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            auroc_final_ave_list = []
            auprc_final_ave_list = []
            number_of_auroc_over_ninety_final = 0
            time = 0
            summary_path = model_folder + "/summary"
            fb = open(summary_path, 'w')
            print("NO\tAUROC\tAUPRC\tAUC>0.9\ttime", file=fb)
            for i in range(fold_number):
                # i = 0
                path_to_save_model = '{}/{}{}.h5'.format(model_folder, dataset, str(i))
                train_file = '{}/train_{}.pickle'.format(data_folder, str(i))
                test_dir = '{}/test_{}'.format(data_folder, str(i))
                total_time, failed, ave_auroc, ave_auprc, number_of_auroc_over_ninety = train.train_main(
                        path_to_save_model, train_file, test_dir, BATCH_SIZE, iteration)
                print('{0}\t{1}\t{2}\t{3}\t{4}'.format(str(i), ave_auroc, ave_auprc, number_of_auroc_over_ninety, total_time), file=fb)
                # record performance
                auroc_final_ave_list.append(ave_auroc)
                auprc_final_ave_list.append(ave_auprc)
                number_of_auroc_over_ninety_final += number_of_auroc_over_ninety
                time += total_time
            auroc_final_ave = round(np.mean(auroc_final_ave_list), 4)
            auprc_final_ave = round(np.mean(auprc_final_ave_list), 4)
            time = int(time/fold_number)
            number_of_auroc_over_ninety_final = int(number_of_auroc_over_ninety_final/fold_number)
            print('summary\t{}\t{}\t{}\t{}'.format(auroc_final_ave, auprc_final_ave, number_of_auroc_over_ninety_final, time), file=fb)
            print("failed number: ", len(failed), file=fb)
            fb.close()
            AUROC.append(auroc_final_ave)
            AUPRC.append(auprc_final_ave)
            over_90.append(number_of_auroc_over_ninety_final)
            Time.append(time)
        except:
            continue
    df = pd.DataFrame({hyper_parameter:value_range, 'AUROC':AUROC, 'AUPRC':AUPRC, '>0.9':over_90, 'Time':Time})
    df = df[[hyper_parameter,'AUROC','AUPRC','>0.9', 'Time']]
    df.sort_values('AUROC', ascending=False, inplace=True)
    df = df.reset_index()
    del df['index'] 
    print(df)
    path = parameter_folder + '/summary'
    df.to_csv(path, sep='\t', index=False)
