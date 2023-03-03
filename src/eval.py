#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from utils import  evaluation_one_target, evaluation_one_dataset, build_new_folder
import pandas as pd

# input_file = '/y/Aurora/Fernie/kinase_project/final_score/DUD-E/scores/aa2ar.score'

def main(input_file=None, input_dir=None, output_dir=None, method_name=None, 
         dataset_name=None):
    if input_dir is not None:
        if len(os.listdir(input_dir))<1:
            print(f"There is no score files in the {input_dir}")
            return
    build_new_folder(output_dir)
    if input_file is not None:
        auc_roc, auc_prc, ef1, ef5, ef10 = evaluation_one_target(input_file)
        df = pd.DataFrame({
            'AUC_ROC': [auc_roc],
            'AUC_PRC': [auc_prc],
            'EF1%': [ef1],
            'EF5%': [ef5],
            'EF10%': [ef10]
            })
        base_name = os.path.basename(input_file)
        print('\n--------------------------------\n'
              f'Performance of {base_name} file\n'
              f'Methods: {method_name}'
        )
        print(df)
        print('--------------------------------\n')
        if output_dir is not None:
            output_file = os.path.join(output_dir, 
                f'{base_name}.{method_name}.performance.xlsx')
            df.to_excel(output_file, sep='\t', index=False)
    if input_dir is not None:
        targets_performance, summary_performance = evaluation_one_dataset(
            input_dir, output_dir, method_name, dataset_name)
        print(f'Methods: {method_name}\n'
              f'Performance of {dataset_name} dataset')
        print(summary_performance)

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input_file', default=None,
        help="The predict result with label file.")
    ap.add_argument('-d', '--input_dir', default=None,
        help="The folder which saves predict results of one dataset.")
    ap.add_argument('-o', '--output_dir', default=None)
    ap.add_argument('-m', '--method_name', default='fernie')
    ap.add_argument('-n', '--dataset_name', required=True)
    args = ap.parse_args()
    main(input_file=args.input_file, 
         input_dir=args.input_dir,
         output_dir=args.output_dir,
         method_name=args.method_name,
         dataset_name=args.dataset_name
         )