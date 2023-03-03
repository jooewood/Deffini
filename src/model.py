#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
import torch.optim.lr_scheduler as lrs
from torchmetrics.functional import auc, accuracy, auroc, average_precision,\
    precision, recall


class FernieModel(pl.LightningModule):
    def __init__(self, 
                 ## hyperparameters for training
                 lr_scheduler=None,
                 lr = 0.0001,
                 weight_decay=0.001,


                 ## hyperparameters for model architecture
                 cf=600, # number of convolutional filters
                 h=100, # number of hidden units
                 datm=200, # size of atom type embedding
                 damino=200, # size of amino acid embedding vector
                 dchrg=200, # size of charge embedding vector
                 ddist=200, # size of distance embedding vector
                 A=35, # size of the atom type dictionary of embeddings
                 D=50, # size of the distance dictionary of embeddings
                 C=50, # size of the charge dictionary of embeddings
                 R=34, # size of the amino acid dictionary of embeddings
                 kc=6, # number of neighboring atoms from compound
                 kp=2, # number of neighboring atoms from protein
                 max_atom_num=100, # The maxium number of atoms in each ligand

                 ## model layer hyperparameters
                 pool_type = 'max',
                 activation='False',
                 dp_rate=0.15,
                 batchnorm=True,
                 **kwargs):
        
        super().__init__()
        # call this to save hyperparameters to the checkpoint
        self.save_hyperparameters()
        
        if dp_rate==0:
            dp_rate = False

        if activation not in ['relu', 'sigmoid', 'tahn',  'leakyrelu', 'gelu', 'elu']:
            self.use_activation = False
        else:
            self.use_activation = True

        self.loss = F.nll_loss
        self.acc = accuracy
        self.average_precision = average_precision
        self.precision = precision
        self.recall = recall

        
        ## hyperparameters for training
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        
        ## hyperparameters for model architecture
        self.cf = cf
        self.h = h 
        self.datm = datm
        self.damino = damino 
        self.dchrg = dchrg 
        self.ddist = ddist 
        self.A = A 
        self.D = D
        self.C = C 
        self.R = R 
        self.kc = kc 
        self.kp = kp 
        self.max_atom_num = max_atom_num
        self.zi = (datm + dchrg + ddist) * (kc +kp) + damino * kp
       
        ## model layer hyperparameters
        self.pool_type = pool_type
        self.activation = activation
        self.batchnorm = batchnorm
        self.dp_rate = dp_rate
        
        self.fc1 = nn.Linear(self.cf, self.h, bias=False)
        self.fc2 = nn.Linear(self.h, 2)
        self.flatten1 = nn.Flatten()

        self.conv1 = nn.Conv2d(1, self.cf, (1, self.zi), 1, bias=False)

        self.embedding_atmo = nn.Embedding(self.A, self.datm)
        self.embedding_chrg = nn.Embedding(self.C, self.dchrg)
        self.embedding_dist = nn.Embedding(self.D, self.ddist)
        self.embedding_amin = nn.Embedding(self.R, self.damino)

        if self.dp_rate:
            self.dp2d = nn.Dropout2d(self.dp_rate)
            self.dp1d = nn.Dropout(self.dp_rate)
            
        if self.use_activation:
            if self.activation=='relu':
                self.activation = nn.ReLU()
            if self.activation=='sigmoid':
                self.activation = nn.Sigmoid()
            if self.activation=='tahn':
                self.activation = nn.Tanh()
            if self.activation=='elu':
                self.activation = nn.ELU()
            if self.activation=='leakyrelu':
                self.activation = nn.LeakyReLU()
            if self.activation=='gelu':
                self.activation = nn.GELU()
                
        if self.batchnorm:
            self.bnm1 = nn.BatchNorm2d(1)
            self.bnm2 = nn.BatchNorm2d(self.cf)
            self.bnm3 = nn.BatchNorm1d(self.cf)
            self.bnm4 = nn.BatchNorm1d(self.h)

        self.convolution1 = nn.Sequential()
        if self.batchnorm:
            self.convolution1.add_module('bnm1', self.bnm1)
        if self.use_activation:
            self.convolution1.add_module('activation1', self.activation)
        if self.dp_rate:
            self.convolution1.add_module('dp2d1', self.dp2d)

        self.convolution1.add_module('conv1', self.conv1)
        
        self.pooling = nn.Sequential()
        if self.batchnorm:
            self.pooling.add_module('bnm2', self.bnm2)
        if self.use_activation:
            self.pooling.add_module('activation2', self.activation)
        if self.dp_rate:
            self.pooling.add_module('dp2d2', self.dp2d)

        if self.pool_type == 'max':
            self.pool1 = nn.MaxPool2d((self.max_atom_num, 1), stride=1)
        elif self.pool_type == 'avg':
            self.pool1 = nn.AvgPool2d((self.max_atom_num, 1), stride=1)
        
        self.pooling.add_module('pool1', self.pool1)


        self.pooling.add_module('flatten1', self.flatten1)
        if self.batchnorm:
            self.pooling.add_module('bnm3', self.bnm3)
        if self.use_activation:
            self.pooling.add_module('activation3', self.activation)
        if self.dp_rate:
            self.pooling.add_module('dp1d1', self.dp1d)
        
        self.fullyconnect1 = nn.Sequential()
        self.fullyconnect1.add_module('fc1', self.fc1)
        # Batchnorm layer
        if self.batchnorm:
            self.fullyconnect1.add_module('bnm4', self.bnm4)
        # Activation layer
        if self.use_activation:
            self.fullyconnect1.add_module('activation4', self.activation)
        # Dropout layer
        if self.dp_rate:
            self.fullyconnect1.add_module('dr1d2', self.dp1d)

    def forward(self, x):
        """
        new version forward
        -----------------------------------------------------------------------
        Embedding layer
        -----------------------------------------------------------------------
        """
        em_atmo = self.embedding_atmo(x[0]).view(-1, self.max_atom_num, 
            (self.kc + self.kp) * self.datm)
        em_chrg = self.embedding_chrg(x[1]).view(-1, self.max_atom_num,
            (self.kc + self.kp) * self.dchrg)
        em_dist = self.embedding_dist(x[2]).view(-1, self.max_atom_num,
            (self.kc + self.kp) * self.ddist)
        em_amin = self.embedding_amin(x[3]).view(-1, self.max_atom_num,
            self.kp * self.damino)

        """
        -----------------------------------------------------------------------
        Cat layer
        -----------------------------------------------------------------------
        """
        out = torch.cat([em_atmo, em_chrg, em_dist, em_amin], 2).view(-1, 1,\
            self.max_atom_num, self.zi)

        """
        -----------------------------------------------------------------------
        First Convoluation layer
        -----------------------------------------------------------------------
        """
        out = self.convolution1(out)
        """
        -----------------------------------------------------------------------
        mask layer
        -----------------------------------------------------------------------
        """
        out = out * x[4]
        """
        -----------------------------------------------------------------------
        Max-pooling layer
        -----------------------------------------------------------------------
        """
        out = self.pooling(out)
        """
        -----------------------------------------------------------------------
        First fully connected layer
        -----------------------------------------------------------------------
        """
        out = self.fullyconnect1(out)
        """
        -----------------------------------------------------------------------
        Output layer (Classifier)
        -----------------------------------------------------------------------
        """
        # fully connected layer
        out = self.fc2(out)
        out = F.log_softmax(out, dim = 1)
        return out

    def classification_metric(self, logits, y, mode):
        loss = self.loss(logits, y)
        preds = torch.exp(logits)
        acc = self.acc(preds, y)
        recall = self.recall(preds, y)
        values = {
            f'{mode}_loss': loss,
            f'{mode}_acc': acc,
            f'{mode}_recall': recall
            }
        return preds[:,1], loss, values

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr,
                    weight_decay=self.weight_decay)
        if self.lr_scheduler is None:
            return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = [batch['atom_type'], 
        #      batch['charge'], 
        #      batch['distance'], 
        #      batch['amino_acid'], 
        #      batch['mask_vector']]
        # y = batch['label'].view(-1)
        logits = self(x)
        _, loss, values = self.classification_metric(logits, y, "train")
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True, 
            logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds, _, values = self.classification_metric(logits, y, "val")
        self.log_dict(values, on_step=True, on_epoch=True, prog_bar=True, 
            logger=True)
        return values

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_recall = torch.stack([x["val_recall"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)
        self.log("ptl/val_recall", avg_recall)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds, _, values = self.classification_metric(logits, y, "test")
        self.log_dict(values, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        logits = self(batch)
        preds = torch.exp(logits)
        return preds[:, 1]

    def train_dataloader(self):
        pass
    
    def val_dataloader(self):
        pass
    
    def test_dataloader(self):
        pass
    
    def predict_dataloader(self):
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("FernieModel")
        ## LR Scheduler
        parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
        parser.add_argument('--lr_decay_steps', default=20, type=int)
        parser.add_argument('--lr_decay_rate', default=0.5, type=float)
        parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

        ## hyperparameters for training
        parser.add_argument('--lr', default=1e-4, type=float)
        parser.add_argument('--weight_decay', default=1e-3, type=float)

        ## hyperparameters for model architecture
        parser.add_argument('--cf', type=int, default=600,
            help="number of convolutional filters")
        parser.add_argument('--h', type=int, default=100,
            help="number of hidden units")
        parser.add_argument('--datm', type=int, default=200,
            help="size of atom type embedding")
        parser.add_argument('--damino', type=int, default=200,
            help="size of amino acid embedding vector")
        parser.add_argument('--dchrg', type=int, default=200,
            help="size of charge embedding vector")
        parser.add_argument('--ddist', type=int, default=200,
            help="size of distance embedding vector")
        parser.add_argument('--A', type=int, default=35,
            help="size of the atom type dictionary of embeddings")
        parser.add_argument('--D', type=int, default=50,
            help="size of the distance dictionary of embeddings")
        parser.add_argument('--C', type=int, default=50,
            help="size of the charge dictionary of embeddings")
        parser.add_argument('--R', type=int, default=34,
            help="size of the amino acid dictionary of embeddings")
        parser.add_argument('--kc', type=int, default=6,
            help="number of neighboring atoms from compound")
        parser.add_argument('--kp', type=int, default=2,
            help="number of neighboring atoms from protein")
        parser.add_argument('--max_atom_num', type=int, default=100,
        help="The maxium number of atoms in each ligand")
        
        ## model layer hyperparameters
        parser.add_argument('--activation', default='False', choices=['relu', 'False'])
        parser.add_argument('--pool_type', choices=['max', 'avg'], default='max')
        parser.add_argument('--dp_rate', type=float, default=0.15)
        parser.add_argument('--batchnorm', default=1, type=int, choices=[0,1],
            help="wether to use batch normlization")
        return parent_parser

# =============================================================================
# from torch_summary import summary
# model = FernieModel()
# summary(model, [(1,100,8), (1,100,8), (1,100,8), (1,100,2), (400, 100, 1)], 
#         dtypes=[torch.LongTensor, torch.LongTensor, torch.LongTensor,
#                 torch.LongTensor, torch.LongTensor], device='cpu')
# =============================================================================
