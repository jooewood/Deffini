#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
from pegaflow.api import File, Directory
from pegaflow.Workflow import Workflow
from main import main_parser
from tqdm import tqdm


class AddTargets(Workflow):
    __doc__ = __doc__
    
    def __init__(self,
                 args=None,
                 direct_run=False,
                 site_handler = 'condor',
                 source_dir = '/y/home/zdx/src/fernie/src',
                 cluster_size=1,
                 ):
        Workflow.__init__(self,
            output_path=args.output_path,
            tmpDir='/tmp/', site_handler=site_handler,
            cluster_size = cluster_size,
            debug = False, needSSHDBTunnel = False, report = False)

        self.num_sanity_val_steps = args.num_sanity_val_steps
        self.cpu_per_trial = args.cpu_per_trial
        self.gpus_per_trial = args.gpus_per_trial

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        ## hyperparameters for training
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        ## hyperparameters for model architecture
        self.cf = args.cf
        self.h = args.h 
        self.datm = args.datm
        self.damino = args.damino 
        self.dchrg = args.dchrg 
        self.ddist = args.ddist 

        ## model layer hyperparameters
        self.pool_type = args.pool_type
        self.activation = args.activation
        self.batchnorm = args.batchnorm
        self.dp_rate = args.dp_rate

        self.test_dir = args.test_dir
        self.source_dir = source_dir
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.valid_file = args.valid_file
        
    def registerExecutables(self):
        Workflow.registerExecutables(self)
        path1 = os.path.join(self.source_dir, "main.py")
        self.registerOneExecutable(path=path1, 
            name="main", clusterSizeMultiplier=1)
        path2 = os.path.join(self.source_dir, "eval.py")
        self.registerOneExecutable(path=path2, 
            name="eval", clusterSizeMultiplier=1)
        
    def registerInputFiles(self):
        self.registerFilesOfInputDir(self.input_dir, 
            input_site_handler=self.input_site_handler)
        self.registerOneInputFile(self.valid_file, 
            input_site_handler=self.input_site_handler)
        self.registerFilesOfInputDir(self.test_dir, 
            input_site_handler=self.input_site_handler)

    def run(self):
        self.setup_run()

        self.registerInputFiles()

        for fold in [0]:
            tmp_input_dir = os.path.join(self.input_dir, str(fold))
            l = len(self.getFilesWithProperSuffixFromFolder(tmp_input_dir,
                '.pickle'))
            for i in tqdm(range(l)):
                train_file = os.path.join(tmp_input_dir, '%s.pickle'%str(i))

                current_output_dir = os.path.join(self.output_dir, str(fold), str(i))
                # current_output_dir_obj = Directory(
                #     directory_type=Directory.LOCAL_STORAGE,
                #     path=os.path.join(self.output_dir, str(fold), str(i))
                # )

                makedirjob = self.addMkDirJob(current_output_dir)

                fitjob = self.addGenericJob(executable=self.main, 
                    extraArgumentList=[
                                    '--working_mode', 'train',
                                    '--training_type', 'holdout',
                                    '--train_file', train_file,
                                    '--valid_file', self.valid_file,
                                    '--test_dir', self.test_dir,
                                    '-o', current_output_dir,
                                    '--batch_size', self.batch_size,
                                    '--num_workers', self.num_workers,
                                    '--weight_decay', self.weight_decay,
                                    '--lr', self.lr,
                                    '--cf',  self.cf,
                                    '--h', self.h,
                                    '--datm', self.datm,
                                    '--damino', self.damino,
                                    '--dchrg', self.dchrg,
                                    '--ddist', self.ddist,
                                    '--pool_type', self.pool_type,
                                    '--activation', self.activation,
                                    '--dp_rate', self.dp_rate,
                                    '--batchnorm', self.batchnorm,
                                    '--num_sanity_val_steps', self.num_sanity_val_steps,
                                    '--debug', self.debug],
                    no_of_cpus=self.cpu_per_trial,
                    # gpu=self.gpus_per_trial,
                    parentJob = makedirjob,
                    transferOutput=True,
                    )
                scores_dir = os.path.join(current_output_dir, 'scores')
                performance_dir = os.path.join(current_output_dir, 'performances')
                # scores_dir = Directory(
                #     directory_type=Directory.LOCAL_STORAGE,
                #     path=os.path.join(current_output_dir, 'scores')
                # )
                # performance_dir = Directory(
                #     directory_type=Directory.LOCAL_STORAGE,
                #     path=os.path.join(current_output_dir, 'performances')
                # )
                self.addMkDirJob(performance_dir)
                evaljob = self.addGenericJob(executable=self.eval,
                    extraArgumentList=[
                        '-d', scores_dir,
                        '-o', performance_dir,
                        '-m', 'fernie',
                        '-n' 'MUV'],
                    parentJob = fitjob,
                    transferOutput=True,
                    )
        self.end_run()

if __name__ == '__main__':
    parser = main_parser()
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    instance = AddTargets(args = args)
    instance.run()

"""
~/src/fernie/src/increase_target_peg_run.py -i /y/Aurora/Fernie/data/structure_based_data/Kernie_feature/increase_targets/Kinase/undersample\
    --output_path dags/Kernie_add_target.yaml\
    --valid_file /y/Aurora/Fernie/data/structure_based_data/MUV_feature/MUV.single_pose_undersampling.pickle\
    --output_dir /y/Aurora/Fernie/EXPS/cf600_h100_dp0.1_False_batch320_down/increase/DUD-E_Kinase_pegaflow

./submit.sh dags/DUD-E_Kinase_add_target.yaml condor 1

pegasus-status -l /y/scratch/zdx/submit/DUD-E_Kinase_add_target.2021.Sep.4T154305
pegasus-analyzer /y/scratch/zdx/submit/DUD-E_Kinase_add_target.2021.Aug.31T112151
pegasus-remove /y/scratch/zdx/submit/DUD-E_Kinase_add_target.2021.Aug.31T112151
"""