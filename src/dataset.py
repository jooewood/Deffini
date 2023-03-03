#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import warnings
warnings.filterwarnings("ignore")

import os
import glob
import torch
import pickle
import numpy as np

from sklearn.model_selection import KFold
from torch import tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl

def collater(data):
        name = []
        atom_type = []
        charge = []
        distance = []
        amino_acid  = []
        mask_vector = []
        label = []
        if 'label' in data[0].keys():
            for unit in data:
                name.append(unit['name'])
                atom_type.append(unit['atom_type'])
                charge.append(unit['charge'])
                distance.append(unit['distance'])
                amino_acid.append(unit['amino_acid'])
                mask_vector.append(unit['mask_vector'])
                label.append(unit['label'])

            return [torch.tensor(atom_type),
                    torch.tensor(charge),
                    torch.tensor(distance),
                    torch.tensor(amino_acid),
                    torch.tensor(mask_vector)], torch.tensor(label)
        else:
            for unit in data:
                name.append(unit['name'])
                atom_type.append(unit['atom_type'])
                charge.append(unit['charge'])
                distance.append(unit['distance'])
                amino_acid.append(unit['amino_acid'])
                mask_vector.append(unit['mask_vector'])
            return [torch.tensor(atom_type), 
                    torch.tensor(charge),
                    torch.tensor(distance),
                    torch.tensor(amino_acid),
                    torch.tensor(mask_vector)]


# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __call__(self, sample):
#         if 'label' in sample.keys():
#             return {
#             "name": sample["name"],
#             "atom_type": tensor(sample["atom_type"], dtype=torch.int64),
#             "charge": tensor(sample["charge"], dtype=torch.int64),
#             "distance": tensor(sample["distance"], dtype=torch.int64),
#             "amino_acid": tensor(sample["amino_acid"], dtype=torch.int64),
#             "mask_vector": tensor(sample["mask_vector"], dtype=torch.float32),
#             "label": tensor(sample["label"], dtype=torch.int64)
#             }
#         else:
#             return {
#             "name": sample["name"],
#             "atom_type": tensor(sample["atom_type"], dtype=torch.int64),
#             "charge": tensor(sample["charge"], dtype=torch.int64),
#             "distance": tensor(sample["distance"], dtype=torch.int64),
#             "amino_acid": tensor(sample["amino_acid"], dtype=torch.int64),
#             "mask_vector": tensor(sample["mask_vector"], dtype=torch.float32)
#             }

class FernieDataset(Dataset):
    def __init__(self, pickle_path=None, cf=600, max_atom_num=100, 
                 transform=None, label=True, debug=False):
        with open(pickle_path, 'rb') as f:
            features = pickle.load(f)
        self.names = features[0]
        self.atom_type = features[1].astype(np.int64)
        self.charge = features[2].astype(np.int64)
        self.distance = features[3].astype(np.int64)
        self.amino_acid = features[4].astype(np.int64)
        self.mask_vector = features[5]
        self.label = label
        if label:
            self.labels = features[6][:,1].astype(np.int64)
        else:
            self.labels = None

        if debug:
            self.names = self.names[:10]
            self.atom_type = self.atom_type[:10]
            self.charge = self.charge[:10]
            self.distance = self.distance[:10]
            self.amino_acid = self.amino_acid[:10]
            self.mask_vector = self.mask_vector[:10]
            if label:
                self.labels = self.labels[:10]

        self.transform = transform
        self.cf = cf
        self.max_atom_num = max_atom_num

    def get_mask_mat(self, i):
        return np.hstack((
                         np.ones((self.cf, i, 1)), 
                         np.zeros((self.cf, self.max_atom_num - i, 1))
                         )).astype(np.float32)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.label:
            sample = {
                "name": self.names[idx],
                "atom_type": self.atom_type[idx],
                "charge": self.charge[idx],
                "distance": self.distance[idx],
                "amino_acid": self.amino_acid[idx],
                "mask_vector": self.get_mask_mat(self.mask_vector[idx]),
                "label": self.labels[idx]
                }
        else:
                sample = {
                "name": self.names[idx],
                "atom_type": self.atom_type[idx],
                "charge": self.charge[idx],
                "distance": self.distance[idx],
                "amino_acid": self.amino_acid[idx],
                "mask_vector": self.get_mask_mat(self.mask_vector[idx])
                }

        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
class FernieDataModule(pl.LightningDataModule):
    def __init__(self, 
                 seed=1234,
                 working_mode='random',
                 training_type='random',
                 n_splits = 10,
                 train_file=None, 
                 valid_file=None, 
                 test_file=None,
                 pred_file=None,
                 validation_split=.1,
                 cf=600,
                 max_atom_num=100,
                 batch_size = 320,
                 num_workers = 12, # number of cores used in dataloader
                 label = True,
                 transform = False,
                 debug=False,
                 **kwargs):
        super().__init__()

        self.debug = debug
        self.validation_split = validation_split
        self.training_type = training_type
        self.seed = seed
        self.working_mode = working_mode
        self.n_splits = n_splits
        if self.working_mode == 'cv':
            self._k_fold = KFold(n_splits=self.n_splits, 
                                 shuffle=True)
        ## input
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.pred_file = pred_file
        
        self.label = label
        self.cf = cf
        self.max_atom_num = max_atom_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def read_dataset(self, file=None, label=True, debug=False):
        if '.pt' == os.path.splitext(os.path.basename(file))[1]:
            return torch.load(file)
        elif '.pickle' == os.path.splitext(os.path.basename(file))[1]:
            return FernieDataset(file,
                    cf=self.cf, max_atom_num=self.max_atom_num,
                    transform=self.transform, label=label, debug=debug)

    def split_train_valid(self, path, seed, validation_split, debug=False):
        if not isinstance(path, list):
            path = [path]

        if len(path) == 1:
            whole_dataset = self.read_dataset(path[0], debug=debug)
        elif len(path) > 1:
            datasets = []
            for file in path:
                datasets.append(self.read_dataset(file, debug=debug))
            whole_dataset = ConcatDataset(datasets)

        valid_set_size = int(validation_split * len(whole_dataset))
        train_set_size = len(whole_dataset) - valid_set_size
        
        self.train_dataset, self.valid_dataset = random_split(whole_dataset, 
            [train_set_size, valid_set_size],  
            generator=torch.Generator().manual_seed(seed))

    def dataset_init(self, mode, files=None, label=True, debug=False):      
        assert mode in ['train', 'valid', 'test', 'pred']
        if mode == 'train':
            label = True
        elif mode == 'valid':
            label = True
        elif mode == 'test':
            label = True
        elif mode == 'pred':
            label = False

        if not isinstance(files, list):
            files = [files]
        if len(files) == 1:
            dataset_ = self.read_dataset(files[0], label, debug=debug)
        elif len(files) > 1:
            datasets = []
            for file in files:
                datasets.append(self.read_dataset(file, label, debug=debug))
            dataset_ = ConcatDataset(datasets)

        if mode == 'train':
            self.train_dataset = dataset_
        elif mode == 'valid':
            self.valid_dataset = dataset_
        elif mode == 'test':
            self.test_dataset = dataset_
        elif mode == 'pred':
            self.predict_dataset = dataset_

    def setup(self, stage: str = None):
        # ------------------
        # Train dataset init
        # ------------------
        if stage == 'fit' or stage == 'validate' or stage is None:
            if self.working_mode == 'train':
                if self.training_type == 'holdout':
                    if self.valid_file is None:
                        print('Do not detect a valid file, program will split'
                              ' validation file by itself.')
                        self.split_train_valid(self.train_file, seed=self.seed, 
                            validation_split = self.validation_split, debug=self.debug)
                    else:
                        self.dataset_init('train', self.train_file, debug=self.debug) 
                        self.dataset_init('valid', self.valid_file, debug=self.debug)
                    if self.pred_file is not None:
                        self.dataset_init('pred', self.pred_file, debug=self.debug)
                elif self.training_type == 'cv':
                    self.dataset_init('valid', self.valid_file, debug=self.debug)
                    dataset_ = []
                    for file in self.train_files:
                        dataset_.append(torch.load(file))
                    self.train_dataset = ConcatDataset(dataset_)
                    if self.pred_file is not None:
                        self.dataset_init('pred', self.pred_file, debug=self.debug)

            elif self.working_mode == 'tune':
                if self.valid_file is not None:
                    self.dataset_init('train', self.train_file, debug=self.debug)
                    self.dataset_init('valid', self.valid_file, debug=self.debug)
                    
                else:
                    self.split_train_valid(self.train_file, seed=self.seed,
                        validation_split=self.validation_split, debug=self.debug)
                # check whether has test file
                if self.test_file is not None:
                    self.dataset_init('test', self.test_file, debug=self.debug)
            
        if stage == 'test' or stage is None:
            self.dataset_init('test', self.test_file, debug=self.debug)
        
        if stage == 'predict' or stage is None:
            self.dataset_init('pred', self.pred_file, debug=self.debug)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                        pin_memory=False, collate_fn=collater,
                        shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                        pin_memory=False, collate_fn=collater,
                        shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                        pin_memory=False, collate_fn=collater,
                        shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size,
                        pin_memory=False, collate_fn=collater,
                        shuffle=False, num_workers=self.num_workers)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FernieDataModule")
        parser.add_argument('--training_type', default='holdout', choices=[
            'holdout', 'cv', 'random'])
        parser.add_argument('--working_mode', default='train', 
            choices=['train', 'pred', 'test', 'valid', 'tune', 'random'])
        parser.add_argument('--batch_size', default=320, type=int)
        parser.add_argument('--num_workers', default=10, type=int)
        parser.add_argument('--seed', default=1234, type=int)
        parser.add_argument('--train_file', default=None, nargs='*')
        parser.add_argument('--valid_file', default=None)
        parser.add_argument('--test_file', default=None, nargs='*')
        parser.add_argument('--pred_file', default=None)
        parser.add_argument('--label', default=True, action='store_false')
        parser.add_argument('--validation_split', default=.1, type=float)
        return parent_parser