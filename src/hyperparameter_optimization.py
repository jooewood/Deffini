#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import warnings
warnings.filterwarnings("ignore")

import os
import time
import pickle
from easydict import EasyDict

# PyTorch
import pytorch_lightning as pl


# Self scripts
from utils import update_args, build_new_folder, get_time_string
from model import FernieModel
from dataset import FernieDataModule
from pytorch_lightning.loggers import TensorBoardLogger

# Ray
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.ax import AxSearch

def tune_by_holdout(args):
    if args.add_time:
        time_string = get_time_string()
        args.output_dir = os.path.join(args.output_dir, f'{time_string}')
    args.log_dir = os.path.join(args.output_dir, 'log')
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ray.init(address='172.22.99.12:6000', _redis_password='5241590000000000')

    """
    ====================== Hyperparameters Optimization =======================    
    """
    start_time = time.time()
    
    def objective_function(config, checkpoint_dir=None):
        config = EasyDict(config)
        model = FernieModel(**config)
        dm = FernieDataModule(seed=args.seed, 
                              working_mode=args.working_mode,
                              training_type=args.training_type,
                              validation_split=args.validation_split,
                              n_splits = args.n_splits,
                              train_file=args.train_file, 
                              valid_file=args.valid_file, 
                              test_file=args.test_file,
                              cf = config.cf, 
                              batch_size = config.batch_size,
                              num_workers = args.num_workers)
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            # If fractional GPUs passed in, convert to int.
            gpus = 1,
            log_every_n_steps = args.log_every_n_steps,
            min_epochs = args.min_epochs,
            progress_bar_refresh_rate=0,

            logger = TensorBoardLogger(
                save_dir = tune.get_trial_dir(), name="", version="."),
            callbacks=[
                TuneReportCallback(
                    {
                        "loss": "ptl/val_loss",
                        "mean_accuracy": "ptl/val_accuracy"
                    },
                    on="validation_end")
            ])
        trainer.fit(model, datamodule=dm)

    config = {
        "weight_decay":  tune.choice([1e-5, 1e-4, 1e-3]), # tune.qloguniform(1e-5, 1e-4, 5e-6)
        "lr":  tune.choice([1e-5, 1e-4, 1e-3]), # tune.qloguniform(1e-5, 1e-4, 5e-6)

        # ------------------
        # hyperparameters for model architecture
        # ------------------
        "cf": tune.choice([600]), 
        "h": tune.choice([100]), 

        # ------------------
        # model layer hyperparameters
        # ------------------
        "pool_type": tune.choice(['max']),
        "activation": tune.choice(['False', 'relu']), 
        "dp_rate":  tune.choice([0.1, 0.15, 0.2]), 
        "batchnorm": tune.choice([1]),
        "batch_size": tune.choice([320, 512, 1024])
    }

    if args.search_alg == 'PBT':
        scheduler = PopulationBasedTraining()
        search_alg = None
    elif args.search_alg == 'bayes':
        search_alg = BayesOptSearch(config)
        scheduler = ASHAScheduler()
    elif args.search_alg == 'hyperopt':
        search_alg = HyperOptSearch()
        scheduler = ASHAScheduler()

    reporter = CLIReporter(
        parameter_columns=["batch_size", "lr", "weight_decay", "cf", "h", "pool_type", 
                            "activation", "dp_rate", "batchnorm"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        objective_function,
        # tune.with_parameters(objective_function, args=args),

        resources_per_trial={
            "cpu": args.cpu_per_trial,
            "gpu": args.gpus_per_trial
        },

        local_dir = args.output_dir,
        metric="mean_accuracy",
        mode="max",
        config=config,

        num_samples=args.num_samples,
        # fail_fast=True,

        search_alg = search_alg,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=args.exp_name)

    print("Best hyperparameters found were: ", analysis.best_config)
    # get all the results of tuning hyper-parameters and save as csv
    result_df = analysis.results_df
    result_df.sort_values(by='mean_accuracy', ascending=False, inplace=True)
    result_df.to_excel(os.path.join(args.output_dir, f'{args.exp_name}_all_trials.xlsx'), index = False)

    analysis_file = os.path.join(args.output_dir, f'{args.exp_name}_analysis.pickle')
    with open(analysis_file, 'wb') as fb:
        pickle.dump(analysis, fb, protocol=4)

    end_time = time.time()
    print((end_time-start_time)/3600, 'hours')