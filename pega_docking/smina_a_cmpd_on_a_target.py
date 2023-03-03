#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import sys
import subprocess
import time
import numpy as np
from urllib import request
import getopt

#Define constant
CPU_NUM = 1
    
def run_smina_single_cmpd(receptor_file, ref_ligand_file, cmpd_file, 
    docking_pose_file, tmp_dir):
    cmpd_file_full_name = os.path.basename(cmpd_file)
    cmpd_id, _ = os.path.splitext(cmpd_file_full_name)
    docking_log_file = os.path.join(tmp_dir, '%s.smina_scores.txt'%cmpd_id)
    try:
        command = " ".join(["smina.static", 
                            "-r", receptor_file, 
                            "-l", cmpd_file,
                            "--autobox_ligand", ref_ligand_file,
                            "--autobox_add", "8",
                            "--exhaustiveness", "16",
                            "--cpu", str(CPU_NUM),
                            "-o", docking_pose_file,
                            "--log", docking_log_file])
        subprocess.call(command, shell=True)
    except:
        print("Docking {0} with {1} failed".format(receptor_file, cmpd_file))
        os.remove(docking_pose_file)

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-r", "--receptor_file", type=str, required=True)
    ap.add_argument("-f", "--ref_ligand_file", type=str, required=True)
    ap.add_argument("-c", "--cmpd_file", type=str, required=True)
    ap.add_argument("-o", "--docking_pose_file", type=str, help="Output bdpqt file", required=True)
    ap.add_argument("-t", "--tmp_dir", type=str, required=True)
    args = ap.parse_args()
    run_smina_single_cmpd(args.receptor_file, args.ref_ligand_file, 
        args.cmpd_file, args.docking_pose_file, args.tmp_dir)    
    
    
