#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from utils import copytree

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-i', '--input_dir', required=True)
    ap.add_argument('-o', '--output_dir', required=True)
    args = ap.parse_args()
    copytree(args.input_dir, args.output_dir)