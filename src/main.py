#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import glob
import pickle
import pandas as pd
import numpy as np

# PyTorch
import torch
from pytorch_lightning import Trainer
import pytorch_lightning as pl

# Self script
from model import FernieModel
from dataset import FernieDataModule
from utils import get_time_string, build_new_folder, update_args
from train import train_by_cv, train_by_holdout
from hyperparameter_optimization import tune_by_holdout


def main_parser(called=False):
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    parser = FernieModel.add_model_specific_args(parser)
    parser = FernieDataModule.add_specific_args(parser)

    parser.add_argument('--debug', default=0, type=int, choices=[1, 0])
    parser.add_argument('--gpu_id', default=0, type=int)
    if called:
        parser.add_argument('-o', '--output_dir')
    else:
        parser.add_argument('-o', '--output_dir', required=True)

    ## Tune hyperparameters
    parser.add_argument('--exp_name')
    parser.add_argument('--num_samples', default=50, type=int)
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--search_alg', default='hyperopt', 
        choices=['bayes', 'PBT', 'ax', 'hyperopt'])
    parser.add_argument('--gpus_per_trial', default=1, type=float)
    parser.add_argument('--cpu_per_trial', default=8, type=float)

    parser.add_argument('--test_dir', default=None)

    ## Predicting Info
    parser.add_argument('-p', '--pred_files', nargs='*', 
                        default=None)
    parser.add_argument('-r', '--results', nargs='*', 
                        default=None)
    parser.add_argument('--add_time', default=False, action='store_true')
    
    ## Early Stopping Info
    parser.add_argument('--es_patience', default=8, type=int)
    parser.add_argument('--es_min_delta', default=0.001, type=float)
    parser.add_argument('--es_monitor', default="ptl/val_accuracy")
    parser.add_argument('--es_mode', default='max')

    ## Model Checkpoint Info
    parser.add_argument('--mc_monitor', default='val_acc')
    parser.add_argument('--mc_save_top_k', default=1)
    parser.add_argument('--mc_mode', default='max')
    parser.add_argument('--checkpoint_path', default=None)

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(num_sanity_val_steps=0)
    parser.set_defaults(check_val_every_n_epoch=1)
    parser.set_defaults(max_epochs=15)
    parser.set_defaults(min_epochs=5)
    # parser.set_defaults(amp_backend='apex')
    #parser.set_defaults(logger=True)
    parser.set_defaults(gpus=1)
    parser.set_defaults(log_every_n_steps=1)
    # parser.set_defaults(auto_select_gpus=True)
    # parser.set_defaults(accelerator="ddp")
    # parser.set_defaults(plugins=DDPPlugin(find_unused_parameters=False))
    #parser.set_defaults(checkpoint_callback=False)
    return parser

def check_file(file):
    if file is not None:
        if not os.path.exists(file):
            print(file, 'not exists.')
            return False
        else:
            return True
    return True

def main(args):
    if args.gpus == 0:
        args.gpus = 0
        print(f"Use CPU")
    else:
        args.gpus = [args.gpu_id]
        print(f"Use GPU {args.gpus}")
    pl.seed_everything(args.seed, workers=True)
    
    if not check_file(args.train_file):
        return
    
    if not check_file(args.valid_file):
        return
    
    if not check_file(args.pred_file):
        return

    if not check_file(args.test_file):
        return

    if not check_file(args.test_dir):
        return
    
    if args.debug:
        args.limit_train_batches = 1
        args.limit_val_batches = 1
        args.fast_dev_run = False
        args.max_epochs = 1
        args.batch_size = 10
        args.num_workers = 10
        args.gpus = None

    if args.working_mode == 'train' and args.train_file is not None:
        if args.training_type == 'cv':
            # ------------
            # Training by cv
            # ------------
            train_by_cv(args)

        elif args.training_type == 'holdout':
            # ------------
            # Training by hold out
            # ------------
            train_by_holdout(args)

    # elif args.working_mode == 'pred' and args.checkpoint_path is not None:
    #     start_time = time.time()
    #     update_args(args, 'pred')
    #     # ------------
    #     # Predicting
    #     # ------------
    #     batch_predict(checkpoint_path=args.checkpoint_path, 
    #                     input_dir=args.test_dir, 
    #                     output_dir=args.results_dir, 
    #                     device=torch.device("cuda:0"))
    #     end_time = time.time()
    #     print((end_time-start_time)/3600, 'hours')

    elif args.working_mode == 'tune':
        tune_by_holdout(args)

if __name__ == "__main__":
    parser = main_parser()
    args = parser.parse_args()
    main(args)
