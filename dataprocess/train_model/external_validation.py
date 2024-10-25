#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import sys
import glob
import predict

def main(data_root, model_path):
    # data_root = '../data/MUV/kinase'
    # model_path = '../final_model/DUD-E.h5'
    type_ = data_root.split('/')[-1]
    out_pre = '../final_score/MUV/scores'
    if not os.path.exists(out_pre):
        os.makedirs(out_pre)
    dataset = model_path.split('/')[-1]
    dataset = dataset.split('.h5')[0]
    if dataset=='kinformation':
        dataset = 'KCD'
    path = data_root + '/*'
    task_folders = glob.glob(path)
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        target = task_folder.split('/')[-1]
        df = predict.test_fn(model_path, task_folder)
        path = '%s/%s.%s.ddk.%s.score' % (out_pre, target, type_, dataset)
        df.to_csv(path, sep='\t', index=False)


main('../data/MUV/kinase', '../final_model/DUD-E.h5')
main('../data/MUV/non_kinase', '../final_model/DUD-E.h5')

main('../data/MUV/kinase', '../final_model/kinformation.h5')
main('../data/MUV/non_kinase', '../final_model/kinformation.h5')  

main('../data/MUV/kinase', '../final_model/DUD-E_non_kinase.h5')
main('../data/MUV/non_kinase', '../final_model/DUD-E_non_kinase.h5')  

main('../data/MUV/kinase', '../final_model/DUD-E_kinase.h5')
main('../data/MUV/non_kinase', '../final_model/DUD-E_kinase.h5')  

main('../data/MUV/kinase', '../final_model/KCD-644.h5')
main('../data/MUV/non_kinase', '../final_model/KCD-644.h5')

main('../data/MUV/kinase', '../final_model/KCD-548.h5')
main('../data/MUV/non_kinase', '../final_model/KCD-548.h5')

main('../data/MUV/kinase', '../final_model/KCD-810.h5')
main('../data/MUV/non_kinase', '../final_model/KCD-810.h5')

main('../data/MUV/kinase', '../final_model/KCD-689.h5')
main('../data/MUV/non_kinase', '../final_model/KCD-689.h5')

main('../data/MUV/kinase', '../final_model/KCD-MUV.h5')
main('../data/MUV/non_kinase', '../final_model/KCD-MUV.h5')

main('../data/MUV/protease', '../final_model/DUD-E_protease.h5')
main('../data/MUV/protease', '../final_model/kinformation.h5')
main('../data/MUV/protease', '../final_model/DUD-E.h5')
main('../data/MUV/protease', '../final_model/DUD-E_non_protease.h5')
main('../data/MUV/protease', '../final_model/DUD-E_kinase.h5')

main('../data/MUV/kinase', '../final_model/DUD-E_protease.h5')
