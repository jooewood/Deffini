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

def main():
    for i in range(3):
        # i = 0
        res = []
        for j in range(359):
            # j = 0
            score_folder = '../add_kinase/%d/MUV_scores/%s' % (i, j)
            files = glob.glob(score_folder+'/*')
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
        res_df.to_csv('../add_kinase/%d.summary' % i)
main()
