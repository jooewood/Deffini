#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import time
from tqdm import tqdm


def a_experiment(suffix, f, extra_argument=None):
    for fold_number in tqdm([0, 1, 2]):
        start=time.time()
        train_path = os.path.join(data_dir, f'train_{fold_number}.pickle')
        valid_path = os.path.join(data_dir, f'test_{fold_number}.pickle')
        test_path = os.path.join(data_dir, f'test_{fold_number}.pickle')
        test_dir = os.path.join(data_dir, f'test_{fold_number}')
        out_path = os.path.join(out_dir, f'fold_{fold_number}', device+suffix)
        if extra_argument is None:
            commond_ = ' '.join([file, batch, works, 
                                '--training_type', 'holdout',
                                '--train_file', train_path, 
                                '--valid_file', valid_path, 
                                '--test_file', test_path, 
                                '--test_dir', test_dir,
                                '-o', out_path])
        else:
            commond_ = ' '.join([file, batch, works,
                                '--training_type', 'holdout',
                                '--train_file', train_path, 
                                '--valid_file', valid_path, 
                                '--test_file', test_path, 
                                '--test_dir', test_dir,
                                '-o', out_path,
                                extra_argument
                                ]
                                )
        os.system(commond_)
        end=time.time()
        f.write(f'Device: {device}{suffix}\n')
        f.write(commond_)
        f.write('\n')
        f.write('Running time: %s Hours\n\n' % str((end-start)/3600))


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input_dir', required=True)
    ap.add_argument('-d', '--device', required=True)
    ap.add_argument('-o', '--output_dir', required=True)
    args = ap.parse_args()

    device = args.device

    file = "./main.py --working_mode train"
    batch = "--batch_size 1024"
    works = "--num_workers 32"
    data_dir = args.input_dir
    out_dir = args.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    run_log_file = os.path.join(out_dir, f'run_log{device}')
    
    with open(run_log_file, "w") as f:
            a_experiment("", f)
            # a_experiment('_01', f, '--amp_level O1')
            # a_experiment('_02', f, '--amp_level O2')
            # a_experiment('_01_16', f, '--amp_level O1 --precision 16')
            # a_experiment('_02_16', f, '--amp_level O2 --precision 16')
            # a_experiment('_16', f, '--precision 16')
