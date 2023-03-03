#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import warnings
warnings.filterwarnings("ignore")

from logging import debug
import os
import time
import glob
from copy import deepcopy
from shutil import copy, rmtree
from os.path import join, exists, basename, dirname, splitext

# Self script
from predict import batch_trainer_predict
from utils import update_args
from dataset import FernieDataModule

# PyTorch
import torch
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import FernieModel

def ExportModelToOnnxandScript(model, dm, args):
    try:
        print("Saving model to onnx.")
        dm.setup('fit')
        train_dataloader = dm.train_dataloader()
        x,y = next(iter(train_dataloader))
        model.set_swish(memory_efficient=False)
        model.to_onnx(args.onnx_path, x, export_params=True)
    except:
        print("Saving model to onnx failed.")
    try:
        print("Saving model to torchscript.")
        script = model.to_torchscript()
        torch.jit.save(script, args.torchscript_path)
    except:
        print("Save model to torchscript failed.")


def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor=args.es_monitor,
        mode=args.es_mode,
        patience=args.es_patience,
        min_delta=args.es_min_delta
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor=args.mc_monitor,
        dirpath=args.checkpoint_dir,
        filename='{epoch}-{val_loss:.5f}-{val_acc:.5f}',
        save_top_k=args.mc_save_top_k,
        mode=args.mc_mode,
        save_last=True
    ))
    return callbacks

def trainer_fit(trainer, model, dm, args=None):
    trainer.fit(model, datamodule=dm)
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    print(f'Best checkpoint is {best_checkpoint_path}')
    ExportModelToOnnxandScript(model, dm, args)
    return best_checkpoint_path, trainer

def pred_afger_train(args, trainer):
    if args.test_file is not None:
        # Testing
        print(f"start to test on {args.test_file}")
        test_result = trainer.test()
        print(test_result)

    if args.test_dir is not None:
        print(f"start to predict on {args.test_dir}")
        batch_trainer_predict(trainer=trainer,
                              input_dir=args.test_dir,
                              output_dir=args.results_dir,
                              label=True,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              cf=args.cf,
                              max_atom_num=args.max_atom_num,
                              debug=args.debug)


def model_init(args):
    print('Model initlizing ...')
    """
    ========================== Initialize model ===============================
    """
    args = update_args(args)
    dict_args = vars(args)
    # print(dict_args)
    # ------------
    # Callbacks
    # ------------
    args.callbacks = load_callbacks(args)
    # ------------
    # Logger
    # ------------
    args.logger = TensorBoardLogger(save_dir=args.log_dir)
    # ------------
    # Initialize model
    # ------------
    model = FernieModel(**dict_args)
    # ------------
    # Initialize trainer
    # ------------
    trainer = Trainer.from_argparse_args(args)
    print("Finished model initlization.")
    return model, trainer, dict_args, args

def train_by_cv(args):
    start_time = time.time()
    """
    ======================= Train by cross validation =========================
    """
    print("Training a dataset by cross validation...")
    output_dir_root = deepcopy(args.output_dir)
    
    file_name = splitext(basename(args.train_file))[0]
    fold_dir = join(dirname(args.train_file), file_name)
    test_files = glob.glob(os.path.join(fold_dir, '*.pt'))
    n_splits = len(test_files)

    if n_splits == 0:
        print(f"There isn't .pt file in the {fold_dir}, please check.")
        return

    print(n_splits, 'fold cross validation')
    for fold, file in enumerate(test_files):
        print(f"Training on fold {fold}...")
        args.output_dir = os.path.join(output_dir_root, f'fold_{fold}')
        model, trainer, _, args = model_init(args)
        sub_files = deepcopy(test_files)
        sub_files.remove(file)
        # ------------
        # DataModule
        # ------------
        dm = FernieDataModule(debug=args.debug,
                              seed=args.seed, 
                              working_mode=args.working_mode,
                              training_type = 'cv',
                              validation_split=args.validation_split,
                              n_splits = args.n_splits,
                              train_files=sub_files,
                              valid_file=file, 
                              cf = args.cf,
                              max_atom_num = args.max_atom_num,
                              batch_size = args.batch_size,
                              num_workers = args.num_workers)
        best_checkpoint_path, trainer = trainer_fit(trainer, 
                                                    model, dm, args=args)
        print(f"Testing on fold {fold}...")
        pred_afger_train(args, trainer)
    end_time = time.time()
    print((end_time-start_time)/3600, 'hours')

def train_by_holdout(args):
    """
    ======================== Traing by hold out ===============================
    """
    start_time = time.time()
    """
    Train by hold out
    """
    model, trainer, dict_args, args = model_init(args)
    print("Training a dataset by hold out...")
    dm = FernieDataModule(**dict_args)
    print(f'Training on {args.train_file}')
    best_checkpoint_path, trainer = trainer_fit(trainer, model, dm, args=args)
    print("Testing...")
    pred_afger_train(args, trainer)
    end_time = time.time()
    print((end_time-start_time)/3600, 'hours')

