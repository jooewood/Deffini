#!/usr/bin/env python2
# -*- coding: future_fstrings -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyflow import WorkflowRunner
import os, sys
import pdb

class DockingSubmit(WorkflowRunner):

    def __init__(self, input_dir, min_index, max_index, src, step, output_dir):
        self.input_dir = input_dir
        self.min_index = min_index
        self.max_index = max_index
        self.src = src
        self.step = step
        self.output_dir = output_dir

    def workflow(self):
        # Create DAG files
        index_list = list(range(self.min_index, self.max_index+1, self.step))
        print("index_list")
        submit_dag_name = 'zero'
        self.addTask(submit_dag_name)
        for i in range(len(index_list)-1):
            tmp_head = index_list[i]
            tmp_end = index_list[i+1]
            dag_name = f"docking_{tmp_head}_{tmp_end}"
            executable = os.path.join(self.src, 'pega_docking.py')
            tmp_dag_file = os.path.join(self.output_dir, "{}.xml".format(dag_name))
            self.addTask(dag_name, f"{executable} -i {self.input_dir} "
                f"-a {tmp_head} -e {tmp_end} -o {tmp_dag_file}",
                dependencies=submit_dag_name)
            submit_dag_name = f"submit_{dag_name}"
            # submit the DAG files created just now
            self.addTask(submit_dag_name, 
                f"./submit.sh {tmp_dag_file} condor",
                dependencies=dag_name)


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("-i", "--input_dir", type=str, required=True)
    ap.add_argument("--min_index", type=int, default=0)
    ap.add_argument("--max_index", type=int, default=480) 
    ap.add_argument("-s", "--step", type=int, default=60)
    ap.add_argument("--src", type=str, default='/y/home/zdx/src/fernie/pega_docking/')
    ap.add_argument("-o", "--output_dir", type=str, required=True)
    ap.add_argument("-b", "--debug", action='store_true', help="Toggle debug value.")
    args = ap.parse_args()
    wflow = DockingSubmit(
        input_dir = args.input_dir,
        min_index = args.min_index,
        max_index = args.max_index,
        src = args.src,
        step = args.step,
        output_dir = args.output_dir
    )
    retval = wflow.run()
    sys.exit(retval)