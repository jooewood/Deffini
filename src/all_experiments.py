#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from glob import glob
from main import main_parser
from main import main as main_train
import time
from eval import main as eval_main
from utils import copytree


def one_fold(fold, dataset, dst, args):
    if args.undersample:
        print('Training by undersample')
        args.train_file = os.path.join(args.data_root, 
                                       f"{dataset}_3_fold/train_{fold}.pickle")
        args.valid_file = os.path.join(args.data_root, 
                                       f"{dataset}_3_fold/test_{fold}.pickle")
        args.test_dir = os.path.join(args.data_root, 
                                       f"{dataset}_3_fold/test_{fold}")
        tmp_output_dir = os.path.join(args.root_output_dir, 
                                       'cv', dataset, f'fold_{fold}')
    else:
        print("Training by raw data.")
        args.train_file = os.path.join(args.data_root, 
                                       f"{dataset}_3_fold/train_{fold}.raw.pickle")
        args.valid_file = os.path.join(args.data_root, 
                                       f"{dataset}_3_fold/test_{fold}.pickle")
        args.test_dir = os.path.join(args.data_root, 
                                       f"{dataset}_3_fold/test_{fold}")
        tmp_output_dir = os.path.join(args.root_output_dir, 
                                       'cv', dataset, f'fold_{fold}')
    args.output_dir = tmp_output_dir
    main_train(args)
    src = os.path.join(tmp_output_dir, 'scores', '*')
    os.system(f'mv {src} {dst}')


def cross_validation(dataset, args):
    print(f'Cross validation on {dataset}')
    final_performance_dir = os.path.join(args.root_output_dir, dataset,
                                         'performances')
    final_scores_dir = os.path.join(args.root_output_dir, dataset,
                                    'scores')
    if not os.path.exists(final_performance_dir):
        os.makedirs(final_performance_dir)
    if not os.path.exists(final_scores_dir):
        os.makedirs(final_scores_dir)
    for i in range(3):
        print(f"{i} fold")
        one_fold(i, dataset, final_scores_dir, args)
    eval_main(input_dir=final_scores_dir, output_dir=final_performance_dir, 
              method_name='fernie', dataset_name=dataset)

def MUV_test(args, filename, dataset, valid_file=None):

    print('Test on MUV, Training on {filename} by')
    args.train_file = os.path.join(args.data_root, 
            f'{dataset}_feature/{filename}.single_pose.pickle')

    args.valid_file = valid_file

    if args.undersample:
        print("by undersample.")
        args.train_file = args.train_file.replace('pose.', 'pose.undersample.')
    else:
        print("by raw training data.")
    tmp_output_dir = os.path.join(args.root_output_dir, 
                                   "MUV", filename)
    args.output_dir = tmp_output_dir
    main_train(args)
    final_performance_dir = os.path.join(tmp_output_dir, 'performances')
    final_scores_dir = os.path.join(tmp_output_dir, 'scores')
    eval_main(input_dir=final_scores_dir, output_dir=final_performance_dir, 
              method_name='fernie', dataset_name="MUV")

def run_allexperiments(args):
    start = time.time()
    args.root_output_dir = os.path.join(args.project_folder,
                                        args.outputfolder,
                                        args.experiment_name)
    
    # Clustered 3-fold cross-validation on DUD-E
    if 0 in args.exps or -1 in args.exps:
        cross_validation("DUD-E", args)

    # Clustered 3-fold cross-validation on Kernie
    if 1 in args.exps or -1 in args.exps:
        cross_validation("Kernie", args)
    
    # Testing on MUV, training by DUD-E subsets
    dataset = "DUD-E"
    args.test_dir = os.path.join(args.data_root, 
                                 "MUV_feature/single_pose")
    if 2 in args.exps or -1 in args.exps:
        # DUD-E
        MUV_test(args, 'DUD-E', dataset, valid_file=args.val_file)
        
    if 3 in args.exps or -1 in args.exps:
        # DUD-E.Kinase
        MUV_test(args, 'DUD-E.Kinase', dataset, valid_file=args.val_file)
        
    if 4 in args.exps or -1 in args.exps:
        # DUD-E.Protease
        MUV_test(args, 'DUD-E.Protease', dataset, valid_file=args.val_file)
    
    if 5 in args.exps or -1 in args.exps:
        # DUD-E.Cytochrome_P450
        MUV_test(args, 'DUD-E.Cytochrome_P450', dataset, valid_file=args.val_file)
        
    if 6 in args.exps or -1 in args.exps:
        # DUD-E.GPCR
        MUV_test(args, 'DUD-E.GPCR', dataset, valid_file=args.val_file)
        
    if 7 in args.exps or -1 in args.exps:
        # DUD-E.Ion_Channel
        MUV_test(args, 'DUD-E.Ion_Channel', dataset, valid_file=args.val_file)
    
    if 8 in args.exps or -1 in args.exps:
        # DUD-E.Miscellaneous
        MUV_test(args, 'DUD-E.Miscellaneous', dataset, valid_file=args.val_file)
        
    if 9 in args.exps or -1 in args.exps:
        # DUD-E.Other_Enzymes
        MUV_test(args, 'DUD-E.Nuclear_Receptor', dataset, valid_file=args.val_file)
    
    if 10 in args.exps or -1 in args.exps:
        # DUD-E.Other_Enzymes
        MUV_test(args, 'DUD-E.Other_Enzymes', dataset, valid_file=args.val_file)
    
    dataset = "Kernie"
    
    if 11 in args.exps or -1 in args.exps:
        # Kernie
        MUV_test(args, 'Kernie', dataset, valid_file=args.val_file)

    if 12 in args.exps or -1 in args.exps:
        # Kernie
        MUV_test(args, 'Kernie-MUV', dataset, valid_file=args.val_file)
    
    end = time.time()
    print((end-start)/3600, 'hours.')

if __name__ == '__main__':
    parser = main_parser(True)
    parser.add_argument('--val_file', 
        default="/y/Aurora/Fernie/data/structure_based_data/"
        "MUV_feature/MUV.single_pose_undersampling.pickle")
    parser.add_argument('--undersample', default=1, type=int, choices=[0,1])
    parser.add_argument('--project_folder', default='/y/Aurora/Fernie')
    parser.add_argument('--data_root', default='/y/Aurora/Fernie/data/structure_based_data')
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--outputfolder', default='EXPS')
    parser.add_argument('--exps', type=int, nargs='+',
        help="-1: All experiments.\n"
             "0: DUD-E CV\n"
             "1: Kernie CV\n"
             "2: Testing on MUV, training by DUD-E\n"
             "3: Testing on MUV, training by DUD-E.Kinase\n"
             "4: Testing on MUV, training by DUD-E.Protease\n"
             "5: Testing on MUV, training by DUD-E.Cytochrome_P450\n"
             "6: Testing on MUV, training by DUD-E.GPCR\n"
             "7: Testing on MUV, training by DUD-E.Ion_Channel\n"
             "8: Testing on MUV, training by DUD-E.Miscellaneous\n"
             "9: Testing on MUV, training by DUD-E.Nuclear_Receptor\n"
             "10: Testing on MUV, training by DUD-E.Other_Enzymes\n"
             "11: Testing on MUV, training by Kernie\n"
             "12: Testing on MUV, training by Kernie without MUV targets\n"
        )
    args = parser.parse_args()
    run_allexperiments(args)

"""
./all_experiments.py --undersample 0 --exps 0 1 --experiment_name max_cf_680_h_292_dp0.1_over --gpu_id 0
"""