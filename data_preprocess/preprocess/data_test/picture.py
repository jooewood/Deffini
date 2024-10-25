#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from skimage import io
import random
font = {'size'   : 14}
matplotlib.rc('font', **font)
labelsize = 14

def add_text(text, ax, x, y):
    y = y/10 + y
    x = x/10 + x
    ax.text(x, y, text, backgroundcolor="white", ha='center', va='top', 
            weight='bold', color='black', fontsize=22, alpha=1)

def sns_boxplot(x, y, data, ymax, text, xtext, ax):
    ymax = ymax/50 + ymax
    a = sns.boxplot(x=x,y=y,data=data, width=0.5, palette="Set3", ax=ax, order=["Smina", "DD"])
    a.set_ylim([0, ymax]) 
    add_text(text, ax, xtext, ymax)
    a.set(xlabel=None)
    a.set_ylabel(y,fontsize=labelsize)

def fig2(): # DDK vs Smina on DUD-E
    dataset = "DUD-E"
    df_ddk = pd.read_table('../final_performance/%s/ddk.performance' % dataset)
    df_ddk['method'] = 'DD'
    df_smina = pd.read_table('../final_performance/%s/smina.performance' % dataset)
    df_smina['method'] = 'Smina'
    df = pd.concat([df_ddk, df_smina])
    fig = plt.figure(figsize=(10, 10), dpi=600)
    axes = fig.subplots(2,2)
    xtext = -0.76
    sns_boxplot("method", "AUC_ROC", df, 1, "A", xtext, axes[0,0])
    sns_boxplot("method", "AUC_PRC", df, 1, "B", xtext, axes[0,1])
    sns_boxplot("method", "EF1%", df,  max(df['EF1%']), "C", xtext, axes[1,0])
    sns_boxplot("method", "EF5%", df,  max(df['EF5%']), "D", xtext, axes[1,1])
    fig.tight_layout()
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig2.png'
    plt.savefig(out)
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig2.eps'
    fig.savefig(out,dpi=600,format='eps')

def fig3(): # DDK vs Smina on kinformation   
    dataset = "kinformation"
    df_ddk = pd.read_table('../final_performance/%s/ddk.performance' % dataset)
    df_ddk['method'] = 'DD'
    df_smina = pd.read_table('../final_performance/%s/smina.performance' % dataset)
    df_smina['method'] = 'Smina'
    df = pd.concat([df_ddk, df_smina])
    fig = plt.figure(figsize=(10, 10), dpi=600)
    axes = fig.subplots(2,2)
    xtext = -0.76
    sns_boxplot("method", "AUC_ROC", df, 1, "A", xtext, axes[0,0])
    sns_boxplot("method", "AUC_PRC", df, 1, "B", xtext, axes[0,1])
    sns_boxplot("method", "EF5%", df,  max(df['EF5%']), "C", xtext, axes[1,0])
    sns_boxplot("method", "EF10%", df,  max(df['EF10%']), "D", xtext, axes[1,1])
    fig.tight_layout()
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig3.png'
    plt.savefig(out)
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig3.eps'
    fig.savefig(out,dpi=600,format='eps')

def draw_lines(X, ylabel, legends, ax, location="lower right"):
    colors = ["royalblue", "brown", "darkorange"]
    colors = colors[0:len(X)]
    ymax = 0; ymin = 1; xmin = 0; xmax = 0
    for x in X:
        if max(x) > ymax: ymax = max(x)
        if min(x) < ymin: ymin = min(x)
        if len(x) > xmax: xmax = len(x)
    ymax = ymax/30 + ymax
    ax.axis([xmin, xmax, ymin, ymax])
    for y, col, lab in zip(X, colors, legends):
        x = range(len(y))
        ax.plot(x, y, c=col, label=lab, marker='s')
    ax.legend(loc=location, title="")
    ax.set_xlabel('Number of kinase targets in training set',fontsize=labelsize)
    ax.set_ylabel('Average %s of kinase targets in test set' % ylabel,fontsize=labelsize)
    return ymax

def con2draw(column_name, dfs, legends, ax, location="lower right"):
    X = []
    for df in dfs:
        X.append(df[column_name])
    X = list(map(list, X))
    ylabel = column_name.split('_mean')[0]
    ymax = draw_lines(X, ylabel, legends, ax, location)
    return ymax
    
def fig4():
    # DUD-E_kinase
    df1 = pd.read_csv('../add_kinase/DUD-E_kinase/MUV_kinase_performance/performance.csv')
    # DUD-E_non_kinase
    df2 = pd.read_csv('../add_kinase/DUD-E_non_kinase/MUV_kinase_performance/performance.csv')
    dfs = [df1, df2]
    xtext = -40
    legends = ['DUD-E_kinase', 'DUD-E_non_kinase']
    fig = plt.figure(figsize=(12, 12), dpi=600)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    column_name = 'AUC_ROC_mean'
    ymax = con2draw(column_name, dfs, legends, ax1, 'upper right') # subplot 1
    column_name = 'AUC_PRC_mean'
    ymax = con2draw(column_name, dfs, legends, ax2, 'upper right') # subplot 2
    column_name = 'EF1%_mean'
    ymax = con2draw(column_name, dfs, legends, ax3, 'upper right') # subplot 3
    column_name = 'EF5%_mean'
    ymax = con2draw(column_name, dfs, legends, ax4, 'upper right') # subplot 4
    fig.tight_layout()
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig4.eps'
    plt.savefig(out,dpi=600,format='eps')
    plt.close()

# def fig5():
#     # DUD-E_kinase
#     df1 = pd.read_csv('../add_kinase/DUD-E_kinase/MUV_kinase_performance/performance.csv')
#     # DUD-E_non_kinase
#     df2 = pd.read_csv('../add_kinase/DUD-E_non_kinase/MUV_kinase_performance/performance.csv')
#     # KCD
#     df3 = pd.read_csv('../add_kinase/KCD/MUV_kinase_performance/performance.csv')
#     dfs = [df1, df2, df3]
#     xtext = -40
#     legends = ['DUD-E_kinase', 'DUD-E_non_kinase', 'KCD']
#     fig = plt.figure(figsize=(12, 12), dpi=600)
#     ax1 = fig.add_subplot(221)
#     ax2 = fig.add_subplot(222)
#     ax3 = fig.add_subplot(223)
#     ax4 = fig.add_subplot(224)
#     column_name = 'AUC_ROC_mean'
#     ymax = con2draw(column_name, dfs, legends, ax1) # subplot 1
#     add_text("A", ax1, xtext, ymax)
#     column_name = 'AUC_PRC_mean'
#     ymax = con2draw(column_name, dfs, legends, ax2, 'upper left') # subplot 2
#     add_text("B", ax2, xtext, ymax)
#     column_name = 'EF1%_mean'
#     ymax = con2draw(column_name, dfs, legends, ax3) # subplot 3
#     add_text("C", ax3, xtext, ymax)
#     column_name = 'EF5%_mean'
#     ymax = con2draw(column_name, dfs, legends, ax4) # subplot 4
#     add_text("D", ax4, xtext, ymax)
#     fig.tight_layout()
#     out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig4.eps'
#     plt.savefig(out,dpi=600,format='eps')
#     plt.close()

def fig5():
    # DUD-E_non_kinase
    df1 = pd.read_csv('../add_kinase/DUD-E_non_kinase/DUD-E_kinase_performance/performance.csv')
    # KCD
    df2 = pd.read_csv('../add_kinase/KCD/DUD-E_kinase_performance/performance.csv')

    dfs = [df1, df2]
    legends = ['DUD-E_non_kinase', 'KCD']
    fig = plt.figure(figsize=(12, 12), dpi=600)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    column_name = 'AUC_ROC_mean'
    xtext = -40
    ymax = con2draw(column_name, dfs, legends, ax1)
    add_text("A", ax1, xtext, ymax)
    column_name = 'AUC_PRC_mean'
    ymax = con2draw(column_name, dfs, legends, ax2)
    add_text("B", ax2, xtext, ymax)
    column_name = 'EF1%_mean'
    ymax = con2draw(column_name, dfs, legends, ax3)
    add_text("C", ax3, xtext, ymax)
    column_name = 'EF5%_mean'
    ymax = con2draw(column_name, dfs, legends, ax4)   
    add_text("D", ax4, xtext, ymax)
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig5.jpg'
    fig.tight_layout()
    plt.savefig(out)
    out = '/home/tensorflow/Desktop/manuscript/CloudStation/Picture/fig5.eps'
    plt.savefig(out,dpi=600,format='eps')
    plt.close()

fig2()
fig3()
fig4()
fig5()
