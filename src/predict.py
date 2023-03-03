#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import warnings
warnings.filterwarnings("ignore")

import glob
import os
import pickle
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from dataset import FernieDataset, FernieDataModule, collater
from model import FernieModel
import pytorch_lightning as pl
from utils import build_new_folder


def pickle_data2df(file, label=True, debug=False):
    with open(file, 'rb') as f:
        features = pickle.load(f)
        if debug:
            features[0] = features[0][:10]
        if label:
            if debug:
                features[6] = features[6][:10]
            return pd.DataFrame({
                'ID': features[0],
                'score': None,
                'label' : features[6][:,1]
                })
        else:
            return pd.DataFrame({
                'ID': features[0],
                'score': None,
                'label': None
                })

def trainer_predict(trainer, pickle_path, output_dir, batch_size, num_workers, 
    label, cf=400, max_atom_num=100, debug=False, model=None):
    print(f'Predicting on {pickle_path}')
    """
    =========================== predict by trainer ============================
    """
    dm = FernieDataModule(debug=debug,
                          working_mode='pred', 
                          pred_file=pickle_path, 
                          cf=cf,
                          max_atom_num=max_atom_num, 
                          batch_size=batch_size, 
                          num_workers=num_workers,
                          label=False)
    if model is not None:
        res = trainer.predict(model=model, datamodule=dm)
    else:
        res = trainer.predict(datamodule=dm, ckpt_path='best')
    preds = []
    for y_hat in res:
        preds.extend(y_hat.data.cpu().numpy().tolist())
    result = pickle_data2df(pickle_path, label=label, debug=debug)
    result['score'] = preds
    result.sort_values(by='score', ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    if output_dir is not None:
        filename = os.path.basename(pickle_path).split('.pickle')[0]
        path_out = os.path.join(output_dir, f'{filename}.score')
        result.to_csv(path_out, index=False, sep='\t')
    return result

def batch_trainer_predict(trainer, pickle_paths, output_dir, label, batch_size,
    num_workers, cf=400, max_atom_num=100, debug=False):
    """
    ======================== Batch predict by trainer =========================
    """
    build_new_folder(output_dir)
    if debug:
        pickle_paths = pickle_paths[:1]
    for pickle_path in tqdm(pickle_paths):
        trainer_predict(debug=debug,
                        trainer=trainer, 
                        pickle_path=pickle_path, 
                        output_dir=output_dir, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        label=label,
                        cf=cf,
                        max_atom_num=max_atom_num)


# if __name__ == "__main__":
#     from argparse import ArgumentParser
#     ap = ArgumentParser()
#     ap.add_argument('-i', '--input_dir', default=None)
#     ap.add_argument('-p', '--pred_file', default=None)
#     ap.add_argument('-c', '--checkpoint_path', required=True)
#     ap.add_argument('-o', '--output_dir', required=True)
#     args = ap.parse_args()
#     if torch.cuda.is_available():
#         if args.input_dir is None and args.predict:
#             batch_trainer_predict(
#                 trainer=,
#                 input_dir=, 
#                 output_dir, 
#                 label, 
#                 batch_size,
#                 num_workers, 
#                 cf=400, 
#                 max_atom_num=100, 
#                 debug=False)
#     else:
#         print('Do you have GPU?')


"""
---------------------------
test
---------------------------
"""
# pickle_path = '/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature/DUD-E.Ion Channel.undersample.pickle'
# output_dir = '/y/Aurora/Fernie/debug/tmp_debug_trainer_predict'
# batch_size = 1024
# num_workers = 20
# model = FernieModel()
# trainer = pl.Trainer(gpus=1, max_epochs=1, check_val_every_n_epoch=1, log_every_n_steps=1)
# dm = FernieDataModule(train_file=pickle_path,
#                       working_mode='train',
#                       training_type='holdout')
# trainer.fit(model, datamodule=dm)

# trainer_predict(trainer=trainer, 
#                 pickle_path=pickle_path, 
#                 output_dir=output_dir, 
#                 batch_size=batch_size, 
#                 num_workers=num_workers, 
#                 label=True)

# batch_trainer_predict(
#     trainer= trainer,
#     input_dir="/y/Aurora/Fernie/data/structure_based_data/DUD-E_3_fold/test_0",
#     output_dir="/y/Aurora/Fernie/debug/tmp_batch_pred_trainer",
#     label=True,
#     batch_size=batch_size, 
#     num_workers=num_workers
# )