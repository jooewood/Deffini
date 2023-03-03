#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from glob import glob
from multiprocessing import Pool
from cv_split import kfold_dataset
from functools import partial


def main(input_dirs, n_splits=10):
    files = []
    for input_dir in input_dirs:
        files = files + glob(os.path.join(input_dir, '*.pickle'))

    for file in files:
        kfold_dataset(file, n_splits)


main(['/y/Aurora/Fernie/data/structure_based_data/DUD-E_feature',
      '/y/Aurora/Fernie/data/structure_based_data/Kernie_feature'
      ])