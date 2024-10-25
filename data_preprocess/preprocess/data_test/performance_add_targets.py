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
import matplotlib
font = {'size'   : 12}
matplotlib.rc('font', **font)
labelsize = 14

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
    ef1 = enrichment(a_pos, l,  0.01)
    ef5 = enrichment(a_pos, l,  0.05)
    ef10 = enrichment(a_pos, l,  0.1)
    #ef15 = enrichment(a_pos, l,  0.15)
    return ef1, ef5, ef10

def assessment(file):
    # file = '/home/tensorflow/kinase_project/final_score/MUV/kinase/548.ddk.DUD-E.score'
    df = pd.read_csv(file)
    df = df.sort_values(by='score', ascending=False)
    df = df.reset_index(drop=True)
    pred = df[['label', 'score']].values
    precision, recall, _ = precision_recall_curve(pred[:, 0], pred[:, 1])
    auc_roc, auc_prc = roc_auc_score(pred[:, 0], pred[:, 1]), auc(recall, precision)
    ef1, ef2, ef3 = enrichment_factor(df)
    return round(auc_roc,4), round(auc_prc, 4), round(ef1, 4), round(ef2, 4), round(ef3, 4)
 
def compute_ave_std(df):
    # df = df_DUD_E_kinase.copy()
    df = df[['AUC_ROC', 'AUC_PRC', 'EF1%', 'EF5%', 'EF10%']]
    res = list(df.apply(np.mean))
    std = list(df.apply(np.std))
    res = res + std
    res = list(map(lambda x: round(x,4), res))
    return res

def get_performance_of_one_scores_folder(path):
    res = []
    for i in range(len(glob.glob(path + '/*'))):
        # i = 0
        folder = path + '/%d' % (i+1)
        files = glob.glob(folder+'/*')
        auc_roc_list = []
        auc_prc_list = []
        EF1_list = []
        EF2_list = []
        EF3_list = []
        for file in files:
            # file = files[0]
            auc_roc, auc_prc, ef1, ef2, ef3 = assessment(file)
            auc_roc_list.append(auc_roc); auc_prc_list.append(auc_prc)
            EF1_list.append(ef1); EF2_list.append(ef2); EF3_list.append(ef3)
        df = pd.DataFrame({'AUC_ROC':auc_roc_list,
                           'AUC_PRC':auc_prc_list,
                           'EF1%':EF1_list,
                           'EF5%':EF2_list,
                           'EF10%':EF3_list
                           })
        res.append(compute_ave_std(df))
    res_df = pd.DataFrame(res)
    res_df.columns = ['AUC_ROC_mean', 'AUC_PRC_mean', 'EF1%_mean', 'EF5%_mean',
                      'EF10%_mean', 'AUC_ROC_std', 'AUC_PRC_std', 'EF1%_std', 
                      'EF5%_std', 'EF10%_std']
    tmp = path.split('/')[-1]
    tmp = tmp.split('_sco')[0]
    result_folder = os.path.dirname(path) + '/%s_performance' % tmp
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    res_df.to_csv(result_folder+'/performance.csv')
    return res_df

def draw_performance(df, path):
    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    ax1.plot(df['AUC_ROC_mean'])
    ax1.set_ylabel("AUC_ROC",fontsize=labelsize)
    ax2.plot(df['AUC_PRC_mean'])
    ax2.set_ylabel("AUC_PRC",fontsize=labelsize)
    ax3.plot(df['EF1%_mean'])
    ax3.set_ylabel("EF1%",fontsize=labelsize)
    ax4.plot(df['EF5%_mean'])
    ax4.set_ylabel("EF5%",fontsize=labelsize)
    fig.tight_layout()
    tmp = path.split('/')[-1]
    tmp = tmp.split('_sco')[0]
    result_folder = os.path.dirname(path) + '/%s_performance' % tmp
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    out = result_folder+'/performance.jpg'
    fig.savefig(out)
    
def main(path):
    # path = '../add_kinase/KCD/DUD-E_kinase_scores'
    df = get_performance_of_one_scores_folder(path)
    draw_performance(df, path)
    
## MUV kinase performance
# DUD-E_non_kinase
main('../add_kinase/DUD-E_non_kinase/MUV_kinase_scores')
# DUD-E_kinase
main('../add_kinase/DUD-E_kinase/MUV_kinase_scores')
# KCD
main('../add_kinase/KCD/MUV_kinase_scores')

## DUD-E kinase performance
# DUD-E_non_kinase
main('../add_kinase/DUD-E_non_kinase/DUD-E_kinase_scores')
# KCD
main('../add_kinase/KCD/DUD-E_kinase_scores')




