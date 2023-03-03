#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from glob import glob
from main import main_parser
import main
import time
from eval import main as eval_main

def seq_increase_targets(args):
    start = time.time()
    
    
    if args.dataset=='Kernie' and args.family!='Kinase':
        args.family='Kinase'
    
    dataset = args.dataset
    undersample = args.undersample
    family = args.family
    output_root = args.output_root
    data_root = args.data_root
    MUV_feature_folder = args.MUV_feature_folder
    valid_file = os.path.join(MUV_feature_folder, f'MUV.{family}.undersample.pickle')
    
    
    if args.test_dir is None:
        if args.test_whole_MUV:
            test_dir = os.path.join(MUV_feature_folder, 'single_pose')
        else:
            if args.single_pose:
                test_dir = os.path.join(MUV_feature_folder, 'single_pose', f'{family}')
            else:
                test_dir = os.path.join(MUV_feature_folder, 'multi_pose', f'{family}')
        args.test_dir = test_dir
    
    if not os.path.exists(args.test_dir):
        print(args.test_dir, 'not exists.')
        return
    
    valid_file = os.path.join(MUV_feature_folder, 
                              f'MUV.{family}.undersample.pickle')
    args.valid_file = valid_file
    if not os.path.exists(valid_file):
        print(valid_file, 'not exists.')
        return
    
    input_dir = os.path.join(data_root, f'{dataset}_feature', 'increase_targets')
    if family is not None:
        input_dir = os.path.join(input_dir, family)
    output_dir = os.path.join(output_root, dataset)
    if family is not None:
        output_dir = os.path.join(output_dir, family)
    
    if not os.path.exists(input_dir):
        print(input_dir, 'not exists.')
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        

    time_ = args.time
    if undersample:
        input_dir = os.path.join(input_dir, 'undersample')
    fold_input_dir = os.path.join(input_dir, str(time_))
    if not os.path.exists(fold_input_dir):
        print(fold_input_dir, 'not exists.')
        return
    sub_pickles = glob(os.path.join(fold_input_dir, '*.pickle'))
    l = len(sub_pickles)
    if args.start is not None and args.end is not None:
        print(f'Start from {args.start} to {args.end}')
        for j in range(args.start, args.end):
            # j = 0
            print("-----------------")
            print(f"{j}th train_file")
            print("-----------------")
            train_file = os.path.join(fold_input_dir, str(j+1)+'.pickle')
            if not os.path.exists(train_file):
                print(train_file, 'not exists.')
            args.train_file = train_file
            tmp_output_dir = os.path.join(output_dir, str(time_), str(j))
            args.output_dir = tmp_output_dir
            main.main(args)
            scores_dir = os.path.join(tmp_output_dir, 'scores')
            performance_dir = os.path.join(tmp_output_dir, 'performances')
            eval_main(input_dir=scores_dir, output_dir=performance_dir, dataset_name="MUV")
    else:
        for j in range(l):
            # j = 0
            print("-----------------")
            print(f"{j}th train_file")
            print("-----------------")
            train_file = os.path.join(fold_input_dir, str(j+1)+'.pickle')
            if not os.path.exists(train_file):
                print(train_file, 'not exists.')
            args.train_file = train_file
            tmp_output_dir = os.path.join(output_dir, str(time_), str(j))
            args.output_dir = tmp_output_dir
            main.main(args)
            scores_dir = os.path.join(tmp_output_dir, 'scores')
            performance_dir = os.path.join(tmp_output_dir, 'performances')
            eval_main(input_dir=scores_dir, output_dir=performance_dir, dataset_name="MUV")
    end = time.time()
    print((end-start)/3600, 'hours.')


if __name__ == '__main__':
    parser = main_parser(True)
    parser.add_argument('--start', default=None,type=int)
    parser.add_argument('--end', default=None, type=int)
    parser.add_argument('--undersample', default=1, type=int, choices=[0,1])
    parser.add_argument('--test_whole_MUV', default=1, type=int, choices=[0,1])
    parser.add_argument('--dataset', choices=['Kernie', 'DUD-E'], 
                        default='Kernie')
    parser.add_argument('--family', choices=['Kinase', 'Protease', 'Nuclear'], 
                        default='Kinase')
    parser.add_argument('--data_root', 
        default='/y/Aurora/Fernie/data/structure_based_data/')
    parser.add_argument('--output_root', 
        default='/y/Aurora/Fernie/debug/add_targets/')
    parser.add_argument('--MUV_feature_folder',
        default='/y/Aurora/Fernie/data/structure_based_data/MUV_feature')
    parser.add_argument('--single_pose', type=int, default=1, choices=[0,1])
    parser.add_argument('--time', type=int, choices=[0,1,2,3,4])
    args = parser.parse_args()
    seq_increase_targets(args)

"""
./increase_targets_seq_run.py --gpu_id 0 --dataset DUD-E --family Kinase --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase

./increase_targets_seq_run.py --gpu_id 0 --dataset DUD-E --family Kinase --time 1 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase

./increase_targets_seq_run.py --gpu_id 0 --dataset DUD-E --family Kinase --time 2 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase

./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase

./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --time 1 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase

./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --time 2 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase

./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase



split train


done n11-0 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 340 --end 358 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n11-0 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 320 --end 340 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie



done n11-0 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 295 --end 305 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie --num_workers 10
done n11-0 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 285 --end 295 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie --num_workers 10
done n101  ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 305 --end 315 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n101  ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 315 --end 320 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie



failed n11-1 ./increase_targets_seq_run.py --gpu_id 1 --dataset Kernie --start 300 --end 320 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
failed n11-1 ./increase_targets_seq_run.py --gpu_id 1 --dataset Kernie --start 280 --end 300 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie

done n101 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 260 --end 280 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n101 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 240 --end 260 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie

done n102 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 220 --end 240 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n102 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 200 --end 220 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n102 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 180 --end 200 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n102 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 160 --end 180 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
done n102 ./increase_targets_seq_run.py --gpu_id 0 --dataset Kernie --start 140 --end 160 --time 0 --output_root /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase_spilt/Kernie
"""