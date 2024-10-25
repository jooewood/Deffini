#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================


import os
import glob
import train

def main(dataset, test_dir=None):
    root_path = os.path.dirname(os.getcwd())
    train_file = root_path + '/final_model/%s.pickle' % dataset
    if not os.path.exists(train_file):
        print("%s not exists." % train_file)
        return 
    path_to_save_model = root_path + '/final_model/%s.h5' % dataset
    train.train_main(path_to_save_model, train_file, test_dir=None, BATCH_SIZE = 1024, iteration = 11)
    
if __name__ == "__main__":
    main("DUD-E")
    main("kinformation")
    main("DUD-E_non_kin")
    main("DUD-E_kinase")
    main("KCD-644")
    main("KCD-548")
    main("KCD-810")
    main("KCD-689")
    main("KCD-MUV")
    main("DUD-E_protease")
    main("DUD-E_non_protease")
