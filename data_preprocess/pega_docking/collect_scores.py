#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import numpy as np
import os


#Basic docking with smina
def parse_docking_file(docking_log_file):
    best_energy = np.nan
    with open(docking_log_file) as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            if line.find("-----+")==0 and lines[i + 1].find("1 ")==0:
               best_energy=float(lines[i + 1].strip().split()[1])
               break
    return best_energy

def main(input_files, output_file):
    with open(output_file, 'w') as fout:
        fout.write("#Cmpd_ID\tBest_energy\n")
        for docking_log_file in input_files:
            cmpd_id = os.path.basename(docking_log_file).split('.')[0]
            best_energy = parse_docking_file(docking_log_file)
            print("%s\t%.4f" % (cmpd_id, best_energy), file=fout)
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-o", dest='output_file', help="Collected scores into a file")
    ap.add_argument('input_files', nargs='+')
    args = ap.parse_args()
    main(args.input_files, args.output_file)