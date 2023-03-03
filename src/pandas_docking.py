#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import pandas as pd
from utils import parallelize_dataframe




def run_docking(line):
    task_folder = os.path.join(line['input_dir'], line['target_name'], line['conformation'])
    receptor_file = os.path.join(task_folder, line['target_path'])
    ref_ligand_file = os.path.join(task_folder, line['ref_ligand'])
    cmpd_file = os.path.join(task_folder, line['cmpd_library'], line['cmpd_type'], line['cmpd_file'])
    docking_pose_file = os.path.join(task_folder, line['docking_pdbqt'], line['cmpd_type'], line['cmpd_file'])
    tmp_dir = os.path.join(task_folder, 'tmp')
    print(' '.join(['./smina_a_cmpd_on_a_target.py',
                    '-r', receptor_file, 
                    '-f', ref_ligand_file, 
                    '-c', cmpd_file,
                    '-o', docking_pose_file,
                    '-t', tmp_dir]))
    if not os.path.exists(docking_pose_file):
        os.system(' '.join(['./smina_a_cmpd_on_a_target.py',
                        '-r', receptor_file, 
                        '-f', ref_ligand_file, 
                        '-c', cmpd_file,
                        '-o', docking_pose_file,
                        '-t', tmp_dir]))
    else:
        sz = os.path.getsize(docking_pose_file)
        if not sz:
            os.system(' '.join(['./smina_a_cmpd_on_a_target.py',
                            '-r', receptor_file, 
                            '-f', ref_ligand_file, 
                            '-c', cmpd_file,
                            '-o', docking_pose_file,
                            '-t', tmp_dir]))
    return "success"


def apply_docking(df):
    df['docking_result'] = df.apply(run_docking, axis=1)
    return df

def parall_docking(df):
    df = parallelize_dataframe(df, apply_docking)
    return df


def main(input_file, start, nrows, run, nrows_of_file):
    if nrows_of_file is None:
        nrows_of_file = sum(1 for line in open(input_file))
    print("Number of file:", nrows_of_file-1)
    if start==0:
        skiprows = None
        df = pd.read_csv(input_file, sep='\t', nrows=nrows)
    else:
        skiprows = range(1, start+1)
        if start+nrows >= nrows_of_file:
            print("The last block.")
            nrows = None
        df = pd.read_csv(input_file, sep='\t', skiprows=skiprows, nrows=nrows)
    print(f"Start from {start} line and read {nrows} lines.")
    if run:
        parall_docking(df)

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input_file', default='/y/Aurora/Fernie/Kernie_docking_guideline.tsv')
    ap.add_argument('-s', '--start', type=int, default=0)
    ap.add_argument('-n', '--nrows', type=int, default=None)
    ap.add_argument('--stop', default=True, action='store_false')
    ap.add_argument('--nrows_of_file', default=2503956)
    args = ap.parse_args()
    main(
        input_file = args.input_file, 
        start = args.start,
        nrows = args.nrows,
        run = args.run,
        nrows_of_file = args.nrows_of_file
    )
