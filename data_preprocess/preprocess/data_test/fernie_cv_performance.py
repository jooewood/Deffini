#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# @author: zdx
# =============================================================================

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
    global dataset
    dataset = data
    root_path = '..'
    files_path = glob.glob('%s/final_score/%s/scores/*'%(root_path, dataset))
    target_performance_path = '%s/final_performance/%s/%s.performance' % (root_path, dataset, method)
    summary_path = '%s/final_performance/%s/%s.summary' % (root_path, dataset, method)
    pair = ['label', 'score']
    targets = []
    auc_roc_list = []
    auc_prc_list = []
    EF1_list = []
    EF2_list = []
    EF3_list = []
    for file in files_path:
        # file = files_path[1]
        tmp = file.split('/')[-1]
        target = tmp.split('.score')[0]
        df = pd.read_table(file)
        df = df.sort_values(by='score', ascending=False)
        df = df.reset_index(drop=True)
        pred = df[pair].values
        precision, recall, _ = precision_recall_curve(pred[:, 0], pred[:, 1])
        auc_roc, auc_prc = roc_auc_score(pred[:, 0], pred[:, 1]), auc(recall, precision)
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
    main("DUD-E", 'ddk')
    main("kinformation", 'ddk')   
    
#def fig1(df):
#    pred = df[pair].values
#    fpr, tpr, ths = metrics.roc_curve(pred[:, 0], pred[:, 1])
#    auc = metrics.auc(fpr, tpr)
#    fig = plt.figure(figsize=(5, 5))
#    ax1 = fig.add_subplot(111)
#    lw = 1.5
#    ax1.plot(fpr, tpr, lw=lw, label=legend+'(AUC=%.3f)' % auc)
#    fig.show()