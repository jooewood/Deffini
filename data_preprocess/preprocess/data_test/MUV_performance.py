#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

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

# def enrichment(df, top):
#     total = len(df)
#     total_active = len(df.query('label==1'))
#     top_number = int(np.around(total * top))
#     top_active = len(df[0:top_number].query('label==1'))
#     return (top_active/top_number)*(total/total_active)
    
def enrichment_factor(df):
    l = len(df)
    a_pos = df[df['label']==1].index
    a_pos = np.array(a_pos) + 1
    ef1 = enrichment(a_pos, l,  0.01)
    ef5 = enrichment(a_pos, l,  0.05)
    ef10 = enrichment(a_pos, l,  0.1)
    #ef15 = enrichment(a_pos, l,  0.15)
    return ef1, ef5, ef10

def assessment(file):
    # file = '/home/tensorflow/kinase_project/final_score/MUV/scores/712.non_kinase.ddk.DUD-E.score'
    df = pd.read_table(file)
    df = df.sort_values(by='score', ascending=False)
    df = df.reset_index(drop=True)
    pred = df[['label', 'score']].values
    precision, recall, _ = precision_recall_curve(pred[:, 0], pred[:, 1])
    auc_roc, auc_prc = roc_auc_score(pred[:, 0], pred[:, 1]), auc(recall, precision)
    ef1, ef2, ef3 = enrichment_factor(df)
    return round(auc_roc,4), round(auc_prc, 4), round(ef1, 4), round(ef2, 4), round(ef3, 4)

# assessment('/home/tensorflow/kinase_project/final_score/MUV/scores/712.non_kinase.ddk.DUD-E_kinase.score')

def ranking_auc(df, mode='T'): # auc, aupr
    """ 'T' from small to big"""
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

def smina(task_folder, method='smina',  mode='T'):
    # task_folder = '/home/tensorflow/kinase_project/data/MUV/kinase/548'
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
    auc_roc, auc_prc = ranking_auc(df, mode=mode)
    ef1, ef2, ef3 = enrichment_factor(df)
    return round(auc_roc,4), round(auc_prc, 4), round(ef1, 4), round(ef2, 4), round(ef3, 4)

def compute_ave_std(df):
    # df = df_DUD_E_kinase.copy()
    l_df = len(df)
    type_ = list(df.family)[0]
    train_set = list(df.train_set)[0]
    df = df[['AUC_ROC', 'AUC_PRC', 'EF1%', 'EF5%', 'EF10%']]
    res = list(df.apply(np.mean))
    std = list(df.apply(np.std))
    res = res + std
    res = list(map(lambda x: round(x,4), res))
    res.append(type_)
    res.append(train_set)
    res.append(l_df)
    return res

def DF_T(df, columns):
    # df = res_df_mean_kinase.copy()
    return pd.DataFrame(df.values.T, columns=columns)


def reshape_df(df):
    colmuns = list(df.train_set)
    df = df.drop(columns=['train_set', 'family', 'num_target'], axis=1)
    df = DF_T(df, colmuns)
    df['Metrics'] = ['AUC_ROC', 'AUC_PRC', 'EF1%', 'EF5%', 'EF10%']
    tmp = ['Metrics'] + colmuns
    df = df[tmp]
    return df

def main(data, method):
    # method = 'ddk'
    # data = 'MUV'
    global dataset
    dataset = data
    root_path = '..'
    files_path = glob.glob('%s/final_score/%s/scores/*'%(root_path, dataset))
    target_performance_path = '%s/final_performance/%s/performance.csv' % (root_path, dataset)
    if not os.path.exists('%s/final_performance/%s' % (root_path, dataset)):
        os.makedirs('%s/final_performance/%s' % (root_path, dataset))
    
    targets = []
    type_list = []
    train_set_list = []
    
    auc_roc_list = []
    auc_prc_list = []
    EF1_list = []
    EF2_list = []
    EF3_list = []
    
    kinase_task_folders = glob.glob('../data/MUV/kinase/*')
    protease_task_folders = glob.glob('../data/MUV/protease/*')
    non_kinase_task_folders = glob.glob('../data/MUV/non_kinase/*')
    
    for task_folder in kinase_task_folders:
        # task_folder = kinase_task_folders[0]
        target = task_folder.split('/')[-1]
        targets.append(target); type_list.append('kinase'); train_set_list.append('Smina')
        auc_roc, auc_prc, ef1, ef2, ef3 = smina(task_folder)
        auc_roc_list.append(auc_roc); auc_prc_list.append(auc_prc)
        EF1_list.append(ef1); EF2_list.append(ef2); EF3_list.append(ef3)

    for task_folder in protease_task_folders:
        target = task_folder.split('/')[-1]
        targets.append(target); type_list.append('protease'); train_set_list.append('Smina')
        auc_roc, auc_prc, ef1, ef2, ef3 = smina(task_folder)
        auc_roc_list.append(auc_roc); auc_prc_list.append(auc_prc)
        EF1_list.append(ef1); EF2_list.append(ef2); EF3_list.append(ef3)

    for task_folder in non_kinase_task_folders:
        target = task_folder.split('/')[-1]
        targets.append(target); type_list.append('non_kinase'); train_set_list.append('Smina')
        auc_roc, auc_prc, ef1, ef2, ef3 = smina(task_folder)
        auc_roc_list.append(auc_roc); auc_prc_list.append(auc_prc)
        EF1_list.append(ef1); EF2_list.append(ef2); EF3_list.append(ef3)
  
    for file in files_path:
        # file = files_path[0]
        file_name = file.split('/')[-1]
        obj = file_name.split('.')
        target, type_, train_set = obj[0], obj[1], obj[3]
        targets.append(target); type_list.append(type_); train_set_list.append(train_set)
        auc_roc, auc_prc, ef1, ef2, ef3 = assessment(file)
        auc_roc_list.append(auc_roc); auc_prc_list.append(auc_prc)
        EF1_list.append(ef1); EF2_list.append(ef2); EF3_list.append(ef3)
    df = pd.DataFrame({'target':targets,
                       'family':type_list,
                       'train_set':train_set_list,
                       'AUC_ROC':auc_roc_list,
                       'AUC_PRC':auc_prc_list,
                       'EF1%':EF1_list,
                       'EF5%':EF2_list,
                       'EF10%':EF3_list
                       })
    df = df.sort_values(by='AUC_ROC', ascending=False)
    df = df.reset_index(drop=True)
    df = df.sort_values(by=['family', 'AUC_ROC'], ascending=[True, False])
    df.to_csv(target_performance_path, index=False)
    # df = pd.read_csv('/home/tensorflow/kinase_project/final_performance/MUV/ddk.performance')
    res = []
    df_tmp = df.query('train_set=="DUD-E" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E_kinase" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E_kinase" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E_non_kinase" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E_non_kinase" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))   
    df_tmp = df.query('train_set=="Smina" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="Smina" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-644" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-644" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-548" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-548" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-810" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-810" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-689" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-689" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-MUV" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD-MUV" & family=="non_kinase"')
    res.append(compute_ave_std(df_tmp))  
    df_tmp = df.query('train_set=="DUD-E_protease" & family=="kinase"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E_kinase" & family=="protease"')
    res.append(compute_ave_std(df_tmp)) 
    df_tmp = df.query('train_set=="DUD-E_protease" & family=="protease"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E_non_protease" & family=="protease"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="KCD" & family=="protease"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="Smina" & family=="protease"')
    res.append(compute_ave_std(df_tmp))
    df_tmp = df.query('train_set=="DUD-E" & family=="protease"')
    res.append(compute_ave_std(df_tmp))
    # res_df = read_csv('')
    res_df = pd.DataFrame(res)
    res_df.columns = ['AUC_ROC_mean', 'AUC_PRC_mean', 'EF1%_mean', 'EF5%_mean',
                      'EF10%_mean', 'AUC_ROC_std', 'AUC_PRC_std', 'EF1%_std', 
                      'EF5%_std', 'EF10%_std', 'family', 'train_set', 'num_target']
    
    res_df = res_df[['train_set', 'family', 'AUC_ROC_mean', 'AUC_PRC_mean', 
                     'EF1%_mean', 'EF5%_mean', 'EF10%_mean', 'AUC_ROC_std', 
                     'AUC_PRC_std', 'EF1%_std', 'EF5%_std', 'EF10%_std', 'num_target']]
    
    res_df_mean = res_df[['train_set', 'family', 'AUC_ROC_mean', 'AUC_PRC_mean', 
                          'EF1%_mean', 'EF5%_mean', 'EF10%_mean', 'num_target']].copy()
    res_df_mean = res_df_mean.sort_values(by=['family', 'AUC_ROC_mean'], ascending=[True, False])
    summary_path = '%s/final_performance/%s/mean.csv' % (root_path, dataset)
    res_df_mean.to_csv(summary_path, index=False)
    ## MUV kinase mean result
    res_df_mean_kinase = res_df_mean.query('family=="kinase"')
    res_df_mean_kinase = reshape_df(res_df_mean_kinase)
    res_df_mean_kinase = res_df_mean_kinase[['Metrics', 'Smina', 'DUD-E', 'DUD-E_protease', 'DUD-E_kinase', 'DUD-E_non_kinase', 'KCD', 'KCD-MUV', 'KCD-644', 'KCD-548', 'KCD-689','KCD-810']]
    path = '%s/final_performance/%s/kinase_mean.csv' % (root_path, dataset)
    res_df_mean_kinase.to_csv(path, index=False)
    ## MUV protease mean result
    res_df_mean_kinase = res_df_mean.query('family=="protease"')
    res_df_mean_kinase = reshape_df(res_df_mean_kinase)
    res_df_mean_kinase = res_df_mean_kinase[['Metrics', 'Smina', 'DUD-E_protease', 'DUD-E_non_protease', 'DUD-E', 'DUD-E_kinase', 'KCD']]
    path = '%s/final_performance/%s/protease_mean.csv' % (root_path, dataset)
    res_df_mean_kinase.to_csv(path, index=False)    
    ## MUV non kinase mean result
    res_df_mean_non_kinase = res_df_mean.query('family=="non_kinase"')
    res_df_mean_non_kinase = reshape_df(res_df_mean_non_kinase)
    res_df_mean_non_kinase = res_df_mean_non_kinase[['Metrics', 'Smina', 'DUD-E', 'DUD-E_kinase', 'DUD-E_non_kinase', 'KCD', 'KCD-MUV', 'KCD-644', 'KCD-548', 'KCD-689','KCD-810']]
    path = '%s/final_performance/%s/non_kinase_mean.csv' % (root_path, dataset)
    res_df_mean_non_kinase.to_csv(path, index=False)
    
    res_df_std = res_df[['train_set', 'family', 'AUC_ROC_std', 'AUC_PRC_std', 'EF1%_std', 
                      'EF5%_std', 'EF10%_std', 'num_target']].copy()

    res_df_std = res_df_std.sort_values(by=['family', 'AUC_ROC_std'], ascending=[True, True])
    summary_path = '%s/final_performance/%s/std.csv' % (root_path, dataset)
    res_df_std.to_csv(summary_path, index=False)
    
main('ddk', 'MUV')
