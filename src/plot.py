#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import matplotlib
matplotlib.use("Agg")


import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
font = {'size'   : 14}
matplotlib.rc('font', **font)
labelsize = 14
import numpy as np
import statsmodels.api as sm



lowess = sm.nonparametric.lowess

def add_text(text, ax, x, y):
    y = y/10 + y
    x = x/10 + x
    ax.text(x, y, text, backgroundcolor="white", ha='center', va='top', 
            weight='bold', color='black', fontsize=22, alpha=1)

def draw_lines(y, ylabel, legend, ax, location="lower right", smooth=True,
             frac=2. / 3.):
    x = np.array(range(len(y)), dtype=float)
    ymax = max(y); ymin = 0; xmin = 0; xmax = len(y)
    ymax = ymax/30 + ymax
    ax.axis([xmin, xmax, ymin, ymax])
    ax.plot(x, y, 'o', c="royalblue", label="raw",
            markerfacecolor='white', markersize=5)
    if smooth:
        y_s = lowess(y, x, frac=frac)[:,1]
        ax.plot(x, y_s, c='red', label="lowess smooth", linewidth=4)

    ax.legend(loc=location, title="")
    ax.set_xlabel('Number of targets in training set',fontsize=labelsize)
    ax.set_ylabel('Average %s of targets in test set' % ylabel,fontsize=labelsize)
    return ymax

def con2draw(column_name, df, legend, ax, location="lower right", smooth=True,
             frac=2. / 3.):
    y = list(df[column_name].values)
    ylabel = column_name
    ymax = draw_lines(y, ylabel, legend, ax, location, smooth=smooth, 
                      frac=frac)
    return ymax
    

def scatter_plot(df, legend, output_dir, fig_name, smooth=True, frac=1./3.):
    fig = plt.figure(figsize=(12, 12), dpi=600)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    column_name = 'AUC_ROC'
    xtext = -2
    ymax = con2draw(column_name, df, legend, ax1, smooth=smooth, frac=frac)
    add_text("A", ax1, xtext, ymax)
    column_name = 'AUC_PRC'
    ymax = con2draw(column_name, df, legend, ax2, smooth=smooth, frac=frac)
    add_text("B", ax2, xtext, ymax)
    column_name = 'EF5%'
    ymax = con2draw(column_name, df, legend, ax3, smooth=smooth, frac=frac)
    add_text("C", ax3, xtext, ymax)
    column_name = 'EF10%'
    ymax = con2draw(column_name, df, legend, ax4, smooth=smooth, frac=frac)
    add_text("D", ax4, xtext, ymax)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{fig_name}.png')
    plt.savefig(out)
    out = os.path.join(output_dir, f'{fig_name}.eps')
    fig.savefig(out,dpi=600,format='eps')

def sns_boxplot(x, y, data, ymax, text, xtext, ax, names, type_="box"):
    ymax = ymax/50 + ymax
    if type_=="box":
        a = sns.boxplot(x=x,y=y,data=data, width=0.9, palette="Set3", ax=ax, 
                        order=names)
    elif type_=="vio":
        a = sns.violinplot(x=x,y=y,data=data, width=0.9, palette="Set3", ax=ax, 
                        order=names)
    a.set_ylim([0, ymax]) 
    add_text(text, ax, xtext, ymax)
    a.set(xlabel=None)
    a.set_ylabel(y,fontsize=labelsize)
    
    
def distribution_plot(files, names, output_dir, fig_name, type_='max'): # DDK vs Smina on DUD-E
    assert type_ in ['max', 'vio']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dfs = []
    for name, file in zip(names, files):
        df = pd.read_table(file)
        df['method'] = name
        dfs.append(df)

    df = pd.concat(dfs)
    fig = plt.figure(figsize=(12, 12), dpi=600)
    axes = fig.subplots(2,2)
    xtext = -0.76
    sns_boxplot("method", "AUC_ROC", df, 1, "A", xtext, axes[0,0], names, type_)
    sns_boxplot("method", "AUC_PRC", df, 1, "B", xtext, axes[0,1], names, type_)
    sns_boxplot("method", "EF1%", df,  max(df['EF1%']), "C", xtext, axes[1,0], 
                names, type_)
    sns_boxplot("method", "EF5%", df,  max(df['EF5%']), "D", xtext, axes[1,1], 
                names, type_)
    fig.tight_layout()
    out = os.path.join(output_dir, f'{fig_name}.png')
    plt.savefig(out)
    out = os.path.join(output_dir, f'{fig_name}.eps')
    fig.savefig(out,dpi=600,format='eps')




"""
dataset='DUD-E'
box_polt(os.path.join(fernie_exp_dir, f'{dataset}/performances/{dataset}.fernie.performance'),
     f'/y/Aurora/Fernie/output/{dataset}/Smina/performances/{dataset}.Smina.performance',
     'Deffini', 'Smina',
     figure_output_dir, 'fig2')
dataset = 'Kernie'
box_polt(os.path.join(fernie_exp_dir, f'{dataset}/performances/{dataset}.fernie.performance'),
     f'/y/Aurora/Fernie/output/{dataset}/Smina/performances/{dataset}.Smina.performance',
     'Deffini', 'Smina',
     figure_output_dir, 'fig3')
"""
