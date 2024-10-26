
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
from pegaflow.DAX3 import File
from pegaflow.Workflow import Workflow

class PegaFlow(Workflow):
    def __init__(self,
        task_folder,
        cmpd_lib,
        docking_dir,
        xml_output_path,
        tool = "smina",
        site_handler = 'condor',
        cluster_size = 50
        ):
        #call the parent class first
        Workflow.__init__(self,
            output_path = xml_output_path,
            tmpDir='/tmp/', site_handler=site_handler,
            cluster_size = cluster_size,
            debug = False, needSSHDBTunnel = False, report = False)
        self.task_folder = task_folder
        self.cmpd_lib = cmpd_lib
        self.docking_dir = docking_dir
        self.source_dir = source_dir
        self.tool = tool

    def registerExecutables(self):
        Workflow.registerExecutables(self)
        path1 = os.path.join(self.source_dir, "smina_a_cmpd_on_a_target.py")
        self.registerOneExecutable(path=path1, 
                                   name="smina_a_cmpd", clusterSizeMultiplier=1)
        path2 = os.path.join(self.source_dir, "collect_scores.py")
        self.registerOneExecutable(path=path2, 
                                   name="collect_scores", clusterSizeMultiplier=1)
        
    def registerInputFiles(self):
        target_dir = os.path.join(self.task_folder, "target")
        self.registerFilesOfInputDir(target_dir)
        cmpd_library_dir = os.path.join(self.task_folder, "cmpd_library", 
                                        self.cmpd_lib)
        self.registerFilesOfInputDir(cmpd_library_dir)
    
    def check_input(self):
        receptor_file = os.path.join(self.task_folder, "target/receptor.pdbqt")
        if not os.path.exists(receptor_file):
            print("No receptor file found, quit.")
            return False
        ref_ligand_file = os.path.join(self.task_folder, 
                                       "target/ref_ligand.pdbqt")
        if not os.path.exists(ref_ligand_file):
            print("No ref ligand file found, quit.")
            return False
        cmpd_library_dir = os.path.join(self.task_folder, "cmpd_library", 
                                        self.cmpd_lib)
        if not os.path.exists(cmpd_library_dir):
            print("No such cmpd library [{0}] found, quit".format(self.cmpd_lib))
            return False
        return receptor_file, ref_ligand_file, cmpd_library_dir

    def smina_a_cmpd_lib(self, receptor_file, ref_ligand_file, cmpd_library_dir, 
                         cmpd_library_docking_pose_pdbqt_dir, score_file, 
                         tmp_dir, parentjobs):
        dockingJobs = []
        for cmpd in os.listdir(cmpd_library_dir):
            cmpd_file = os.path.join(cmpd_library_dir, cmpd)
            docking_pose_file = File(os.path.join(cmpd_library_docking_pose_pdbqt_dir, cmpd))
            cmpd_file_full_name = os.path.basename(cmpd_file)
            cmpd_id, _ = os.path.splitext(cmpd_file_full_name)
            energy_output_file = File(os.path.join(tmp_dir, "%s.%s.score"%(cmpd_id, self.tool)))
            dockingJobs.append(self.addGenericJob(executable=self.smina_a_cmpd, 
                    transferOutput=True, no_of_cpus=8,
                    parentJobLs = parentjobs,
                    extraOutputLs = [docking_pose_file, energy_output_file],
                    extraArgumentList=["-r", receptor_file, 
                                       "-f", ref_ligand_file,
                                       "-c", cmpd_file,
                                       "-d", docking_pose_file,
                                       "-t", tmp_dir]
                    ))
        self.addGenericJob(executable=self.collect_scores, 
                    transferOutput=True, 
                    parentJobLs = dockingJobs,
                    extraOutputLs = [score_file],
                    extraArgumentList=["-t", self.docking_dir, 
                                       "-c", self.cmpd_lib]
                    )

        

    def run(self):
        self.setup_run()
        
        ## Define docking pose output
        if not self.check_input():
            return
        else:
            receptor_file, ref_ligand_file, cmpd_library_dir = self.check_input()
            
        CreateFolderJobs = []
        ## Output dir
        task_folder = self.docking_dir
        # Define docking pose output
        cmpd_library_docking_pose_pdbqt_dir = os.path.join(task_folder, "docking_pdbqt", 
                                              self.cmpd_lib)
        CreateFolderJobs.append(self.addMkDirJob(cmpd_library_docking_pose_pdbqt_dir))
        cmpd_library_docking_pose_pdb_dir = os.path.join(task_folder, "docking_pdb",
                                            self.cmpd_lib)
        CreateFolderJobs.append((self.addMkDirJob(cmpd_library_docking_pose_pdb_dir)))
            
        # Define docking score output
        smina_score_dir = os.path.join(task_folder, "scoring_%s"%self.tool)
        CreateFolderJobs.append(self.addMkDirJob(smina_score_dir))
        score_out_dir = os.path.join(task_folder, "_".join(["scoring", self.tool]))
        score_file = File(os.path.join(score_out_dir, ".".join([self.cmpd_lib, self.tool, "tsv"])))
        
        # Define temp folder
        tmp_dir = os.path.join(task_folder, 'tmp', self.cmpd_lib)
        CreateFolderJobs.append(self.addMkDirJob(tmp_dir))
        
        ## Run docking
        self.smina_a_cmpd_lib(receptor_file = receptor_file,
                              ref_ligand_file = ref_ligand_file, 
                              cmpd_library_dir = cmpd_library_dir, 
                              cmpd_library_docking_pose_pdbqt_dir = \
                                  cmpd_library_docking_pose_pdbqt_dir,
                              score_file = score_file, 
                              tmp_dir = tmp_dir,
                              parentjobs = CreateFolderJobs
                              )
        self.end_run()

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-t", "--task_folder", type=str, required=True)
    ap.add_argument("-c", "--cmpd_lib", type=str, required=True)
    ap.add_argument("-d", "--docking_dir", type=str, required=True)
    ap.add_argument("-x", "--xml_output_path", type=str, default='./docking.python.xml')
    ap.add_argument("-s", "--source_dir", type=str, help="Folder of all source codes",
            default='~/src/fernie/pega_docking/')
    args = ap.parse_args()
    instance = PegaFlow(
        task_folder = args.task_folder,
        cmpd_lib = args.cmpd_lib,
        docking_dir = args.docking_dir,
        source_dir = args.source_dir,
        xml_output_path = args.xml_output_path,
        )
    instance.run()
