#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")


import os
import sys
import math
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial
from multiprocessing import Pool
import time
from shutil import copy, rmtree
import shutil
import biotite.sequence.io.fasta as fasta
import biotite.sequence.align as align
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import seaborn as sns
import matplotlib.pyplot as plt


def get_family(names, info, name_col, family_col):
    familys = []
    failed = []
    info[name_col] = [str(x) for x in info[name_col]]
    for name in names:
        # name = names[0]
        try:
            familys.append(info.query('%s=="%s"' % (name_col, name))[family_col].values[0])
        except:
            failed.append(name)
    print(f'Lengths is {len(familys)} ')
    print(f'Failed number: {len(failed)}')
    return familys


def MultiProteinAliganments(file, filename, out_dir=None, dpi=600, 
                            figsize=(10, 30), index_col=0, 
                            tight_layout = True,
                            **kwargs):
    """
    >>> file = '/y/Aurora/Fernie/data/DUD-E_Kernie_MUV.fst'
    >>> out_dir = '/y/Aurora/Fernie/Report'
    >>> filename = 'DUD-E_Kernie_MUV'
    >>> identities = MultiProteinAliganments(file, out_dir)
    """
    figure_path = os.path.join(out_dir, f'{filename}_clustermap.png')
    identities_path = os.path.join(out_dir, f'{filename}_identities.xlsx')
    if not os.path.exists(identities_path):
        names, _  = read_fasta(file)
        fasta_file = fasta.FastaFile()
        fasta_file.read(file)
        
        sequences = list(fasta.get_sequences(fasta_file).values())
        
        # BLOSUM62
        substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()
        # Matrix that will be filled with pairwise sequence identities
        identities = np.ones((len(sequences), len(sequences)))
        # Iterate over sequences
        for i in tqdm(range(len(sequences))):
            for j in range(i):
                # Align sequences pairwise
                alignment = align.align_optimal(
                    sequences[i], sequences[j], substitution_matrix
                )[0]
                # Calculate pairwise sequence identities and fill matrix
                identity = align.get_sequence_identity(alignment)
                identities[i,j] = identity
                identities[j,i] = identity
        identities = pd.DataFrame(identities, index = names, columns = names)
        identities = 1 - identities
    else:
        print("Already have identities matrix.")
        identities = read_df(identities_path, index_col=index_col)

    linkage = hc.linkage(sp.distance.squareform(identities), method='average')
    plt.figure(dpi=dpi, figsize=figsize)
    plot = sns.clustermap(identities, row_linkage=linkage, col_linkage=linkage, 
                   figsize=figsize, yticklabels=True, xticklabels=False, 
                   col_cluster=True, row_cluster=True, **kwargs)
    fig = plot.fig
    if tight_layout:
        fig.tight_layout()
    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.exists(identities_path): 
            identities.to_excel(identities_path)
            print(f"Succeed to save identities matrix into {identities_path}")
        print(f"Saving clustermap figure into {figure_path}")
        plt.savefig(figure_path) 
        plt.close() 
    return identities

def read_fasta(file):
    """
    >>> names, seqs = read_fasta(file)
    """

    with open(file, 'r') as f:
        lines = f.readlines()
    names = []
    seqs = []
    for line in lines:
        line = line.strip()
        if '>' in line:
            names.append(line.replace('>', ''))
        elif len(line) > 0:
            seqs.append(line)
    try:
        assert len(names) == len(seqs)
    except:
        print('Names is not equal to sequences.')
        return
    print(f'There are {len(names)} proteins.')
    return names, seqs

def GetFileSuffix(path):
    """
    >>> suffix = GetFileSuffix(path)
    """
    return os.path.splitext(os.path.basename(path))[1]

def GetFileName(path):
    """
    file_name = GetFileName(path)
    """
    return os.path.splitext(os.path.basename(path))[0]

def read_df(file, index_col=None, header=0):
    """
    >>> df = read_df(file)
    """
    if GetFileSuffix(file) == '.xlsx':
        
        if index_col is not None:
            df = pd.read_excel(file, index_col=index_col)
        else:
            df = pd.read_excel(file)
    else:
        with open(file, 'r') as f:
            line = f.readline().strip()
        if '\t' in line:
            sep = '\t'
        elif ' ' in line:
            sep = ' '
        elif ',' in line:
            sep = ','
        if index_col is not None:
            df = pd.read_excel(file, index_col=index_col)
        else:
            df = pd.read_csv(file, sep=sep, header=header)
    return df


def seq2fasta(seqs, names, out):
    """
    >>> seq2fasta(seqs, names, out)
    """
    with open(out, 'w') as f:
        for name, seq in zip(names, seqs):
            print(f'>{name}', file=f)
            print(seq, file=f)
    print(f'Succeed to write {len(names)} proteins into {out}')

def df2fasta(df, out, seq_col, name_col):
    seqs = list(df[seq_col].values)
    names = list(df[name_col].values)
    seq2fasta(seqs, names, out)

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in tqdm(os.listdir(src)):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# Self script
from model import FernieModel
from dataset import FernieDataModule, FernieDataset, collater
from cv_split import kfold_dataset

def check_file(path):
    if path is None:
        return False
    elif not os.path.exists(path):
        print(f"Please check the path: {path}")
    else:
        return True

def update_args(args, mode='train'):
    if args.add_time:
        time_string = get_time_string()
        args.output_dir = os.path.join(args.output_dir, f'{time_string}')
    args.default_root_dir = args.output_dir
    build_new_folder(args.output_dir)
    if mode == 'train':
        if args.logger:
            args.log_dir = os.path.join(args.output_dir, 'log')
            build_new_folder(args.log_dir)
        args.checkpoint_dir = os.path.join(args.output_dir, 'models')
        build_new_folder(args.checkpoint_dir)
        # weitht
        args.weights_save_path = os.path.join(args.checkpoint_dir, 'weights')
        build_new_folder(args.weights_save_path)
        # ONNX
        args.onnx_path = os.path.join(args.checkpoint_dir, 'fernie.onnx')
        # Torchscript
        args.torchscript_path = os.path.join(args.checkpoint_dir, 
                                             'model_script.pt')
    if check_file(args.test_dir):
        args.results_dir = os.path.join(args.output_dir, 'scores')
        build_new_folder(args.results_dir)
    return args

def delete_a_folder(dir_path):
    try:
        rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))


def build_new_folder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        delete_a_folder(dir_path)
        os.makedirs(dir_path)
        

def display_dict(dict_):
    for key,value in dict_.items():
        print(key, ":", value)

def try_copy(source, target):
    try:
       copy(source, target)
    except IOError as e:
       print("Unable to copy file. %s" % e)
       exit(1)
    except:
       print("Unexpected error:", sys.exc_info())
       exit(1)

def get_time_string():
    time_string = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    return str(time_string)

def ranking_auc(df, ascending=False): # Smina use ascending = True
    """ 'T' from small to big"""
    df.sort_values(by='score', ascending = ascending, inplace=True)
    df.reset_index(drop=True, inplace=True)
    auc_roc, auc_prc=np.nan, np.nan
    l = len(df)
    a_pos = list(df[df['label']==1].index)
    a_pos = np.array(a_pos) + 1
    ##Generate the seq of active cmpd
    a_seen = np.array(range(1,len(a_pos)+1))
    ##Calculate auc contribution of each active_cmpd
    d_seen = a_pos-a_seen
    ##Calculate AUC ROC
    a_total=len(a_pos)
    d_total = l - len(a_pos)
    contri_roc = d_seen/(a_total*d_total)
    auc_roc = 1.0 - np.sum(contri_roc)
    ##Calculate AUC PRC
    auc_prc = (1/a_total)*np.sum((1.0*a_seen)/a_pos)
    return auc_roc, auc_prc


def enrichment(a_pos, total_cmpd_number, top):
    ##Calculate total/active cmpd number at top% 
    top_cmpd_number = math.ceil(total_cmpd_number*top)
    top_active_number = 0
    for a in a_pos:
        if a>top_cmpd_number: break
        top_active_number += 1
    ##Calculate EF
    total_active_number = len(a_pos)
    ef = (1.0*top_active_number/top_cmpd_number)*(\
        total_cmpd_number/total_active_number)
    return ef
    
def enrichment_factor(df, target_col='score', ascending=False):
    df.sort_values(by=target_col, ascending=ascending, inplace=True) # high to low
    df.reset_index(drop=True, inplace=True)
    l = len(df)
    a_pos = df[df['label']==1].index
    a_pos = np.array(a_pos) + 1
    ef1 = enrichment(a_pos, l,  0.01)
    ef5 = enrichment(a_pos, l,  0.05)
    ef10 = enrichment(a_pos, l,  0.1)
    return ef1, ef5, ef10

def evaluation_one_target(file, target_col='score', ascending=False):
    df = pd.read_table(file)
    df.sort_values(by=target_col, ascending=ascending, inplace=True) # high to low
    df.reset_index(drop=True, inplace=True)
    auc_roc, auc_prc = ranking_auc(df)
    ef1, ef5, ef10 = enrichment_factor(df)
    return round(auc_roc,4), round(auc_prc, 4), round(ef1, 4), round(ef5, 4),\
        round(ef10, 4)

def comput_aver(x):
    return round(np.mean(x), 4)

def compute_ave_sd(df, cols=['AUC_ROC', 'AUC_PRC', 'EF1%', 'EF5%', 'EF10%']):
    # df = df_DUD_E_kinase.copy()
    df = df[cols]
    means = [round(x, 4) for x in list(df.apply(np.mean))]
    sds = [round(x, 4) for x in list(df.apply(np.std))]
    result = {}
    for col, mean, sd in zip(cols, means, sds):
        result[col]=[f'{mean}±{sd}']
    return pd.DataFrame(result)

def evaluation_one_dataset(input_dir, output_dir, method_name, dataset_name):
    files_path = glob.glob(os.path.join(input_dir,'*'))
    targets = []
    auc_roc_list = []
    auc_prc_list = []
    EF1_list = []
    EF2_list = []
    EF3_list = []
    print(f'Evaluating each target predict result in the {dataset_name} '
          f'dataset, predicted by {method_name}.')
    for file in tqdm(files_path):
        if '.score' not in file:
            continue
        # file = files_path[29]
        tmp = file.split('/')[-1]
        target = tmp.split('.score')[0]
        auc_roc, auc_prc, ef1, ef5, ef10 = evaluation_one_target(file)
        targets.append(target)
        auc_roc_list.append(auc_roc)
        auc_prc_list.append(auc_prc)
        EF1_list.append(ef1)
        EF2_list.append(ef5)
        EF3_list.append(ef10)
    targets_performance = pd.DataFrame({'target':targets,
                       'AUC_ROC':auc_roc_list,
                       'AUC_PRC':auc_prc_list,
                       'EF1%':EF1_list,
                       'EF5%':EF2_list,
                       'EF10%':EF3_list
                       })
    target_performance_path = os.path.join(output_dir, 
        f'{dataset_name}.{method_name}.performance')
    targets_performance.to_csv(target_performance_path, sep='\t', index=False)
    summary_performance = compute_ave_sd(targets_performance)
    summary_path = os.path.join(output_dir, 
        f'{dataset_name}.{method_name}.performance.summary')
    summary_performance.to_csv(summary_path, sep='\t', index=False)
    return targets_performance, summary_performance

"""
=========================== parallelize apply =================================
"""

def parallelize_dataframe(df, func, **kwargs):
    CPUs = multiprocessing.cpu_count()

    num_partitions = CPUs - 1 # number of partitions to split dataframe
    num_cores = CPUs - 1 # number of cores on your machine

    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    func = partial(func, **kwargs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def judge_whether_is_dir(path):
    if not os.path.isdir(path):
        return False
    else:
        return True

def remain_path_of_dir(x):
    return list(filter(judge_whether_is_dir, x))
