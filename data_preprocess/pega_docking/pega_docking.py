#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import glob
import os
from pegaflow.DAX3 import File
from pegaflow.Workflow import Workflow
from tqdm import tqdm

class PegaFlow(Workflow):
    def __init__(self,
        input_dir,
        task_index_start,
        task_index_stop,
        source_dir,
        dir_depth,
        output_path,
        tool = "smina",
        site_handler = 'condor',
        cluster_size = 50
        ):
        #call the parent class first
        Workflow.__init__(self,
            output_path = output_path,
            tmpDir='/tmp/', site_handler=site_handler,
            cluster_size = cluster_size,
            debug = False, needSSHDBTunnel = False, report = False)
        self.input_dir = input_dir
        self.task_index_start = task_index_start
        self.task_index_stop = task_index_stop
        self.source_dir = source_dir
        self.dir_depth = dir_depth
        self.tool = tool

    def remain_items_that_are_dir(self, x):
    	def judge_whether_is_dir(path):
    		if not os.path.isdir(path):
    			return False
    		else:
    			return True
    	return list(filter(judge_whether_is_dir, x))

    def registerExecutables(self):
        Workflow.registerExecutables(self)
        path1 = os.path.join(self.source_dir, "smina_a_cmpd_on_a_target.py")
        self.registerOneExecutable(path=path1, 
            name="smina_a_cmpd", clusterSizeMultiplier=1)
        path2 = os.path.join(self.source_dir, "collect_scores.py")
        self.registerOneExecutable(path=path2, 
            name="collect_scores", clusterSizeMultiplier=0)
        
    def registerInputFiles(self, task_folder, cmpd_lib):
        target_dir = os.path.join(task_folder, "target")
        self.registerFilesOfInputDir(target_dir)
        cmpd_library_dir = os.path.join(task_folder, "cmpd_library", 
                                        cmpd_lib)
        self.registerFilesOfInputDir(cmpd_library_dir)
    
    def check_input(self, task_folder, cmpd_lib):
        print("  Checking input files ... ", flush=True, end="")
        receptor_file = os.path.join(task_folder, "target/receptor.pdbqt")
        if not os.path.exists(receptor_file):
            print("No receptor file found, quit.")
            return False
        ref_ligand_file = os.path.join(task_folder, 
                                       "target/ref_ligand.pdbqt")
        if not os.path.exists(ref_ligand_file):
            print("No ref ligand file found, quit.")
            return False
        cmpd_library_dir = os.path.join(task_folder, "cmpd_library", 
                                        cmpd_lib)
        if not os.path.exists(cmpd_library_dir):
            print("No such cmpd library [{0}] found, quit".format(cmpd_lib))
            return False
        print("  Done.", flush=True)
        return receptor_file, ref_ligand_file, cmpd_library_dir

    def smina_a_cmpd_lib(self, task_folder, cmpd_lib, docking_dir, parentjobs):
        print(f" Working on {cmpd_lib} ...", flush=True)
        ## Define docking pose output
        try:
            receptor_file, ref_ligand_file, cmpd_library_dir = self.check_input(task_folder, cmpd_lib)
        except:
            return
            
        CreateFolderJobs = parentjobs
        ## Output dir
        task_folder = docking_dir
        # Define docking pose output
        cmpd_library_docking_pose_pdbqt_dir = os.path.join(task_folder, "docking_pdbqt", 
                                              cmpd_lib)
        CreateFolderJobs.append(self.addMkDirJob(cmpd_library_docking_pose_pdbqt_dir))
        cmpd_library_docking_pose_pdb_dir = os.path.join(task_folder, "docking_pdb",
                                            cmpd_lib)
        CreateFolderJobs.append((self.addMkDirJob(cmpd_library_docking_pose_pdb_dir)))
            
        # Define docking score output
        smina_score_dir = os.path.join(task_folder, "scoring_%s"%self.tool)
        CreateFolderJobs.append(self.addMkDirJob(smina_score_dir))
        score_out_dir = os.path.join(task_folder, "_".join(["scoring", self.tool]))
        score_file = File(os.path.join(score_out_dir, ".".join([cmpd_lib, self.tool, "tsv"])))
        
        # Define temp folder
        tmp_dir = os.path.join(task_folder, 'tmp', cmpd_lib)
        CreateFolderJobs.append(self.addMkDirJob(tmp_dir))
        
        collect_score_job = self.addGenericJob(executable=self.collect_scores, 
            transferOutput=True, 
            outputFile = score_file,
            outputArgumentOption= "-o"
            )
            
        for cmpd in tqdm(os.listdir(cmpd_library_dir)):
            cmpd_file = os.path.join(cmpd_library_dir, cmpd)
            docking_pose_file = File(os.path.join(cmpd_library_docking_pose_pdbqt_dir, cmpd))
            cmpd_file_full_name = os.path.basename(cmpd_file)
            cmpd_id, _ = os.path.splitext(cmpd_file_full_name)
            docking_log_file = File(os.path.join(tmp_dir, '%s.smina_scores.txt'%cmpd_id))
            docking_job = self.addGenericJob(executable=self.smina_a_cmpd, 
                transferOutput=True, no_of_cpus=4, outputFile=docking_pose_file,
                outputArgumentOption="-o",
                parentJobLs = CreateFolderJobs,
                extraOutputLs = [docking_log_file],
                extraArgumentList=["-r", receptor_file, 
                                    "-f", ref_ligand_file,
                                    "-c", cmpd_file,
                                    "-t", tmp_dir]
            )
            # addInputToMergeJob() also adds docking_job as a parent of mergeJob.
            self.addInputToMergeJob(mergeJob=collect_score_job, inputF=docking_log_file,
                inputArgumentOption="", parentJobLs=[docking_job])

    def run(self):
        self.setup_run()
        begining_jobs = []
        dir_depth = self.dir_depth
        input_dir = self.input_dir
        dataset_name = input_dir.split('/')[-1]
        dataset_docking_dir = dataset_name
        begining_jobs.append(self.addMkDirJob(dataset_docking_dir))
        for i in range(dir_depth):
            input_dir = os.path.join(input_dir, '*')
        task_folders = self.remain_items_that_are_dir(glob.glob(input_dir))
        task_folders = task_folders[self.task_index_start:self.task_index_stop]
        no_of_tasks = len(task_folders)
        for i, task_folder in enumerate(task_folders):
            print(f"Working on {i}/{no_of_tasks}:{task_folder} ... ", flush=True)
            docking_dir = os.path.join(dataset_docking_dir, '/'.join(task_folder.split('/')[-dir_depth:]))
            for cmpd_lib in ['active', 'decoy']:
                self.smina_a_cmpd_lib(task_folder, cmpd_lib, docking_dir, begining_jobs)
        self.end_run()

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--input_dir", type=str, required=True)
    ap.add_argument("-a", "--task_index_start", type=int, required=True)
    ap.add_argument("-e", "--task_index_stop", type=int, required=True)
    ap.add_argument("-s", "--source_dir", type=str, help="Folder of all source codes",
        default='/y/home/zdx/src/fernie/pega_docking/')
    ap.add_argument("-d", "--dir_depth", type=int, default=2,
        help="Depth of the input folder, where the input data is located.")
    ap.add_argument("-o", "--output_path", type=str, required=True,
        help="The output file that will contain the pegasus DAG")
    args = ap.parse_args()
    instance = PegaFlow(
        input_dir = args.input_dir,
        task_index_start = args.task_index_start,
        task_index_stop = args.task_index_stop,
        source_dir = args.source_dir,
        dir_depth = args.dir_depth,
        output_path = args.output_path
        )
    instance.run()
