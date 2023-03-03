#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import pandas as pd
import os
from rdkit_utils import SMILES2mol2InChI
from argparse import ArgumentParser

def main(input_dir,output_dir):
    for file in os.listdir(input_dir):
        df = pd.read_csv(os.path.join(input_dir,file), sep='\t')
        print("start to convert to InchI")
        InChI_df = SMILES2mol2InChI(df)
        print("finish convert to InchI")
        InChI_df.sort_values(by='label',ascending=False)
        print("start to drop NaN")
        InChI_df.dropna()
        print(len(InChI_df))
        print("start to drop duplicates")
        print(len(InChI_df))
        InChI_df.drop_duplicates(subset=None, keep='first', inplace=True)
        print("finish drop duplicates")
        print(len(InChI_df))
        outputfile_name = os.path.splitext(os.path.basename(file))[0]         
        outfile = os.path.join(output_dir, f'{outputfile_name}.csv')         
        InChI_df.to_csv(outfile, index=False, sep='\t')
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Path to targets CSV file')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Path to the result of merge test targets CSV file')
    args = parser.parse_args()
    main(input_dir=args.input_dir, 
         output_dir=args.output_dir)