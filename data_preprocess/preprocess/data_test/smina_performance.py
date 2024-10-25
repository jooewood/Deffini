#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

import os
import glob
import pandas as pd
import numpy as np

def ranking_auc(df, mode='T'):
    if mode=='T':
        df = df.sort_values(by='score', ascending = True)
        df = df.reset_index(drop=True)
    else:
        df = df.sort_values(by='score', ascending = False)
        df = df.reset_index(drop=True)
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
    top_cmpd_number = np.around(total_cmpd_number*top)
    top_active_number = 0
    for a in a_pos:
        if a>top_cmpd_number: break
        top_active_number += 1
    ##Calculate EF
    total_active_number = len(a_pos)
    ef = (1.0*top_active_number/top_cmpd_number)*(total_cmpd_number/total_active_number)
    return ef

def enrichment_factor(df):
    l = len(df)
    a_pos = df[df['label']==1].index
    a_pos = np.array(a_pos) + 1
    ef5 = enrichment(a_pos, l,  0.05)
    ef10 = enrichment(a_pos, l,  0.1)
    ef15 = enrichment(a_pos, l,  0.15)
    if dataset=="DUD-E":
        ef1 = enrichment(a_pos, l,  0.01)
        return ef1, ef5, ef10
    else:
        return ef5, ef10, ef15

def main(data, method):
    # data = "DUD-E"
    # method = "smina"
    global dataset
    dataset = data
    root_path = '..'
    dataroot = '%s/data/%s' % (root_path, dataset)
    if dataset=="DUD-E":
        task_folders = glob.glob(dataroot+'/*')
    elif dataset=='kinformation':
        task_folders = glob.glob(dataroot+'/*/*')
    target_performance_path = '%s/final_performance/%s/%s.performance' % (root_path, dataset, method)
    summary_path = '%s/final_performance/%s/%s.summary' % (root_path, dataset, method)
    targets = []
    auc_roc_list = []
    auc_prc_list = []
    EF1_list = []
    EF2_list = []
    EF3_list = []
    for task_folder in task_folders:
        # task_folder = task_folders[0]
        if dataset=='DUD-E':
            target = task_folder.split('/')[-1]
        elif dataset=="kinformation":
            target = '/'.join([task_folder.split('/')[-2], task_folder.split('/')[-1]])
        if dataset=="kinformation":
            out = task_folder+'/scoring_{0}/docking.{0}.tsv'.format(method)
            if not os.path.exists(out):
                df_ac = pd.read_table(task_folder+'/scoring_{0}/active.{0}.tsv'.format(method))
                df_de = pd.read_table(task_folder+'/scoring_{0}/decoy.{0}.tsv'.format(method))
                df_m = pd.concat([df_ac, df_de])
                df_m = df_m.sort_values(by=['Best_energy'])
                df_m = df_m.rename(columns={'#Cmpd_ID':'#Pose_ID', 'Best_energy':'smina_score'})
                df_m.to_csv(out, sep='\t', index=False)
                df_ac['Label'] = 1
                df_de['Label'] = 0
                df_m = pd.concat([df_ac, df_de])
                df_m = df_m[['#Cmpd_ID', 'Label']]
                if not os.path.exists(task_folder + '/answer'):
                    os.makedirs(task_folder + '/answer')
                out = task_folder+'/answer/docking.answer.tsv'
                df_m.to_csv(out, sep='\t', index=False)
        answer_path = task_folder+'/answer/docking.answer.tsv'
        if not os.path.exists(answer_path):
            print('%s not exists.' % answer_path)
            return
        pred_path = task_folder + '/scoring_{0}/docking.{0}.tsv'.format(method)
        if not os.path.exists(pred_path):
            print('%s not exists.' % pred_path)
            return
        pred_df = pd.read_table(pred_path)
        real_df = pd.read_table(answer_path)
        df = pd.merge(pred_df, real_df, left_on="#Pose_ID", right_on="#Cmpd_ID")
        del df['#Cmpd_ID']
        df.rename(columns={'smina_score':'score', 'Label':'label'}, inplace=True)
        df = df.sort_values(by='score', ascending=True)
        df = df.reset_index(drop=True)
        auc_roc, auc_prc = ranking_auc(df, mode='T')
        ef1, ef2, ef3 = enrichment_factor(df)
        targets.append(target)
        auc_roc_list.append(auc_roc)
        auc_prc_list.append(auc_prc)
        EF1_list.append(ef1)
        EF2_list.append(ef2)
        EF3_list.append(ef3)
    if dataset=='DUD-E':
        df = pd.DataFrame({'target':targets,
                           'AUC_ROC':auc_roc_list,
                           'AUC_PRC':auc_prc_list,
                           'EF1%':EF1_list,
                           'EF5%':EF2_list,
                           'EF10%':EF3_list
                           })
        res = pd.DataFrame(columns=['AUC_ROC', 'AUC_PRC', 'EF1%', 'EF5%', 'EF10%'])
    elif dataset=="kinformation":
        df = pd.DataFrame({'target':targets,
                           'AUC_ROC':auc_roc_list,
                           'AUC_PRC':auc_prc_list,
                           'EF5%':EF1_list,
                           'EF10%':EF2_list,
                           'EF15%':EF3_list,
                           })
        res = pd.DataFrame(columns=['AUC_ROC', 'AUC_PRC', 'EF5%', 'EF10%', 'EF15%'])
    df.to_csv(target_performance_path, sep='\t', index=False)
    def comput_aver(x):
        return round(np.mean(x), 4)
    res.loc[0] = df.iloc[:, 1:6].apply(np.mean, result_type='expand')
    res.to_csv(summary_path, sep='\t', index=False)

if __name__ == "__name__":
    main("DUD-E", 'smina')
    main("kinformation", 'smina')
