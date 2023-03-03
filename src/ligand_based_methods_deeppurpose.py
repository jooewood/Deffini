#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from DeepPurpose import CompoundPred as models
from DeepPurpose import utils
import pandas as pd
from glob import glob
from eval import main as eval_main
from utils import build_new_folder

for drug_encoding in ['CNN', 'Transformer', "Morgan"]:
    target_encoding = None
    
    if drug_encoding=='CNN' and target_encoding=='CNN':
        method_name = 'GanDTI'
    elif target_encoding is None:
        method_name = drug_encoding
    
    print('drug_encoding', drug_encoding)
    print('target_encoding', target_encoding)
    
    undersample = True
    
    output_dir = os.path.join('/y/Aurora/Fernie/EXPS/', method_name)
        
    data_root = '/y/Aurora/Fernie/data/structure_based_data/smiles_seq'
    cuda_id = 0
    num_workers = 10
    train_epoch = 20
    
    
    valid_file = None 
    
    assert drug_encoding in ["CNN",
                             "MPNN",
                             "Transformer",
                             "Morgan",
                             "ESPF",
                             "Pubchem",
                             "rdkit_2d_normalized",
                             "Daylight",
                             "CNN_RNN",
                             "ErG",
                             "DGL_GCN",
                             "DGL_NeuralFP",
                             "DGL_AttentiveFP",
                             "DGL_GIN_AttrMasking",
                             "DGL_GIN_ContextPred"]
    
    assert target_encoding in [None, 
                                "AAC",
                                "PseudoAAC",
                                "Conjoint_triad",
                                "Quasi-seq",
                                "ESPF",
                                "CNN",
                                "CNN_RNN",
                                "Transformer"
                                ]

    
    def preprocess_data(file, drug_encoding, split_method='no_split', 
        frac = [0.8, 0.1, 0.1], random_seed = 'TDC', xcols='SMILES', 
        ycols='label', target_encoding=None, pro_col="protein"):
        data = pd.read_csv(file, sep='\t')
        data.dropna(inplace=True)
        X = data[xcols].values
        y = data[ycols].values
        if target_encoding is not None:
            X_protein = data[pro_col].values
        
        if split_method == "no_split":
            if target_encoding is not None:
                train = utils.data_process(X, X_protein, y, 
                    drug_encoding = drug_encoding,
                    target_encoding = target_encoding,
                    split_method=split_method)
            else:
                train = utils.data_process(X_drug = X, y = y, 
                    drug_encoding = drug_encoding,
                    split_method=split_method)
            return train
        else:
            if target_encoding is not None:
                train, val, test = utils.data_process(X, X_protein, y,
                    drug_encoding = drug_encoding,
                    target_encoding = target_encoding,
                    split_method=split_method, 
                    frac = frac,
                    random_seed = random_seed)
            else:
                train, val, test = utils.data_process(X_drug=X, y=y, 
                    drug_encoding = drug_encoding,
                    split_method=split_method, 
                    frac = frac,
                    random_seed = random_seed)
            return train, val, test

    def get_test_result(y_pred, file, out_dir):
        df = pd.read_csv(file, sep='\t')
        df.dropna(inplace=True)
        name = os.path.splitext(os.path.basename(file))[0]
        df['score'] = y_pred
        path = os.path.join(out_dir, f'{name}.score')
        df = df[['ID', 'score', 'label']]
        df.sort_values('score', ascending=False, inplace=True)
        df.to_csv(path, index=False, sep='\t')
        return df
    
    
    def train_test(out_dir, train_file, dst=None, valid_file=None, test_file=None, 
                   test_dir=None, cuda_id=0, num_workers=10, train_epoch=20, 
                   target_encoding=None):
        print(f'Training on {train_file}')
        if valid_file is None:
            # Training set and Validation set
            train, val, test = preprocess_data(train_file, drug_encoding, 
                'random', target_encoding=target_encoding)
        else:
            if not os.path.exists(valid_file):
                print(valid_file, 'not exists.')
                return
            else:
                # Training set
                train = preprocess_data(train_file, drug_encoding,
                    target_encoding=target_encoding)
                # Validation set
                val = preprocess_data(valid_file, drug_encoding,
                    target_encoding=target_encoding)

                if test_file is None:
                    test = preprocess_data(valid_file, drug_encoding,
                                           target_encoding=target_encoding)
                else:
                    test = preprocess_data(test_file, drug_encoding,
                                           target_encoding=target_encoding)

        if not os.path.exists(os.path.join(out_dir, 'scores')):
            os.makedirs(os.path.join(out_dir, 'scores'))
        if not os.path.exists(os.path.join(out_dir, 'performances')):
            os.makedirs(os.path.join(out_dir, 'performances'))
        build_new_folder(os.path.join(out_dir, 'models'))

        config = utils.generate_config(drug_encoding = drug_encoding, 
                                       target_encoding=target_encoding,
                                 result_folder = os.path.join(out_dir, 'tmp'),
                                 cuda_id = cuda_id,
                                 num_workers = num_workers,
                                 train_epoch = train_epoch)
        
        model = models.model_initialize(**config)
        model.train(train, val, test)
        model.save_model(os.path.join(out_dir, 'models'))
    
        test_files = glob(os.path.join(test_dir, '*.csv'))
        try:
            for file in test_files:
                    X_test = preprocess_data(file, drug_encoding)
                    y_pred = model.predict(X_test)
                    get_test_result(y_pred, file, os.path.join(out_dir, 'scores'))
            print(f'Finished prediction after training on {train_file}')
            src = os.path.join(out_dir, 'scores', '*')
            if dst is not None:
                os.system(f'mv {src} {dst}')
            return "Success"
        except:
            print("Failed to predict after training")
            return train_file
    
    def cross_validation(dataset, data_root, output_dir, target_encoding=None):
        dst_scores = os.path.join(output_dir, dataset, 'scores')
        if not os.path.exists(dst_scores):
            os.makedirs(dst_scores)
        dst_performances = os.path.join(output_dir, dataset, 'performances')
        if not os.path.exists(dst_performances):
            os.makedirs(dst_performances)
    
        print(f"Cross validation on {dataset}")
        for i in range(3):
            print(f"Training on fold {i}")
            # i = 0
            if undersample:
                train_file = os.path.join(data_root, f'{dataset}/train_{i}.undersample.csv')
            else:
                train_file = os.path.join(data_root, f'{dataset}/train_{i}.csv')
            tmp_output_dir = os.path.join(output_dir, 'cv', dataset, f'fold_{i}')
            test_dir = os.path.join(data_root, dataset, f'test_{i}')
            train_test(tmp_output_dir, train_file, dst=dst_scores, test_dir=test_dir,
                cuda_id=cuda_id, num_workers=num_workers, train_epoch=train_epoch,
                target_encoding=target_encoding
                )
        if len(os.listdir(dst_scores)) <1:
            print(f'There is no score files in {dst_scores}')
            return
        eval_main(input_dir=dst_scores, output_dir=dst_performances, 
            method_name=method_name, dataset_name=dataset)

    def MUV_test(file_name, out_dir, dataset, target_encoding=None):
        print(f"Testing on MUV, training by {file_name}")
        train_file = os.path.join(data_root, dataset, f'{file_name}.csv')
        test_dir = '/y/Aurora/Fernie/data/ligand_based_data/MUV'
        if undersample:
            train_file = train_file.replace('.csv', '.undersample.csv')
        if not os.path.exists(train_file):
            print(train_file, 'not exists.')
            return
        tmp_output_dir = os.path.join(out_dir, "MUV", file_name)
        if not os.path.exists(tmp_output_dir):
            os.makedirs(tmp_output_dir)
        train_test(tmp_output_dir, train_file, test_dir=test_dir, 
                   target_encoding=target_encoding)
        dst_scores = os.path.join(tmp_output_dir, 'scores')
        if len(os.listdir(dst_scores)) <1:
            print(f'There is no score files in {dst_scores}')
            return
        eval_main(input_dir=dst_scores,
                  output_dir=os.path.join(tmp_output_dir, 'performances'), 
            method_name=method_name, dataset_name="MUV")
    
    # cross_validation('DUD-E', data_root, output_dir, target_encoding=target_encoding)
    # cross_validation('Kernie', data_root, output_dir, target_encoding=target_encoding)

    dataset = "DUD-E"
    MUV_test('DUD-E_Nuclear_Receptor', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_Kinase', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_Protease', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_Cytochrome_P450', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_GPCR', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_Ion_Channel', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_Miscellaneous', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('DUD-E_Other_Enzymes', output_dir, dataset, target_encoding=target_encoding)
    
    # dataset = "Kernie"
    # MUV_test('Kernie', output_dir, dataset, target_encoding=target_encoding)
    # MUV_test('Kernie-MUV', output_dir, dataset, target_encoding=target_encoding)
