#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:56:04 2018

@author: Edwin
"""

import numpy as np
import pandas as pd
import pymc3 as pm


def glm_mcmc_inference(df, formula, family, I):
    """
    Calculates the Markov Chain Monte Carlo trace of
    a Generalised Linear Model Bayesian linear regression
    model on supplied data.

    df: DataFrame containing the data
    formula: Regressing equation in terms of columns of DataFrame df
    family: Type of liner model. Takes a pymc object (pm.glm.families).
    I: Number of iterations for MCMC

    """
    # Use PyMC3 to construct a model context
    basic_model = pm.Model()
    with basic_model:
        # Create the glm using the Patsy model syntax
        pm.glm.GLM.from_formula(str(formula), df, family=family)
        start = pm.find_MAP()
        step = pm.NUTS()

        trace = pm.sample(I, step, start, progressbar=False)

        return(trace)

def build_permutation(p, arr):
    new = [0 for i in range(len(p))] 
    l = len(p)
    base = min(p)
    for i in range(l):
            new[i] = arr[p[i]-base]
    return(new)
'''
def create_blocks(df1, df2):
    #blocks = np.unique(df1['block'])
    df1 = df1.sort_values(by = ['block']).reset_index(drop = True)
    df2 = df2.sort_values(by = ['block']).reset_index(drop = True)
    print(df1)
    print(df2)
    n1 = len(df1)
    n2 = len(df2)
    i = 0
    if n1 > n2:
        k = 1
        while i < len(df2):
            if df1['block'][i] != df2['block'][i]:
                df2 = pd.concat([df2[0:i], pd.DataFrame(np.full((1, len(df2.columns)), np.nan), columns = list(df2.columns)), df2[i:]]).reset_index(drop = True)
                df2['block'][i] = df1['block'][i]
            i = i + 1
        df2 = pd.concat([df2, pd.DataFrame(np.full((n1-i, len(df2.columns)), np.nan), columns = list(df2.columns))]).reset_index(drop = True)
        df2.loc[i:,'block'] = df1.loc[i:,'block']
    else:
        k = 2
        while i < len(df1):
            if df2['block'][i] != df1['block'][i]:
                df1 = pd.concat([df1[0:i], pd.DataFrame(np.full((1, len(df1.columns)), np.nan), columns = list(df1.columns)), df1[i:]]).reset_index(drop = True)
                df1['block'][i] = df2['block'][i]
            else:
                i = i + 1
        df1 = pd.concat([df1, pd.DataFrame(np.full((n2-i, len(df2.columns)), np.nan), columns = list(df2.columns))]).reset_index(drop = True)
        df1.loc[i:,'block'] = df2.loc[i:,'block']
    return([df1, df2])'''
    
def create_blocks(df1, df2):
    #blocks = np.unique(df1['block'])
    df1 = df1.sort_values(by = ['block']).reset_index(drop = True)
    df2 = df2.sort_values(by = ['block']).reset_index(drop = True)

    if df2['block'][0] != df1['block'][0]:
        if df2['block'][0] < df1['block'][0]:
            df1 = pd.concat([pd.DataFrame(np.full((1, len(df1.columns)), np.nan), columns = list(df1.columns)), df1]).reset_index(drop = True)
            df1['block'][0] = df2['block'][0]
        else:
            df2 = pd.concat([pd.DataFrame(np.full((1, len(df2.columns)), np.nan), columns = list(df2.columns)), df2]).reset_index(drop = True)
            df2['block'][0] = df1['block'][0]
    n1 = len(df1)
    n2 = len(df2)
    current_block = df2['block'][0]
    i = 1
    while i < min(n1, n2):
        if df2['block'][i] != current_block:
            if df1['block'][i] == current_block:
                df2 = pd.concat([df2[0:i], pd.DataFrame(np.full((1, len(df2.columns)), np.nan), columns = list(df2.columns)), df2[i:]]).reset_index(drop = True)
                df2['block'][i] = df1['block'][i]
                n2 = n2 + 1
            else:
                current_block = df2['block'][i]
                i = i + 1
        elif df1['block'][i] != current_block:
            df1 = pd.concat([df1[0:i], pd.DataFrame(np.full((1, len(df1.columns)), np.nan), columns = list(df1.columns)), df1[i:]]).reset_index(drop = True)
            df1['block'][i] = df2['block'][i]
            n1 = n1 + 1
        else:
            i = i + 1
    if n1 > n2:
        df2 = pd.concat([df2, pd.DataFrame(np.full((n1-i, len(df2.columns)), np.nan), columns = list(df2.columns))]).reset_index(drop = True)
        df2.loc[i:,'block'] = df1.loc[i:,'block']
    elif n1 < n2:
        df1 = pd.concat([df1, pd.DataFrame(np.full((n2-i, len(df2.columns)), np.nan), columns = list(df2.columns))]).reset_index(drop = True)
        df1.loc[i:,'block'] = df2.loc[i:,'block']
    
    blocks = [[0,0] for i in range(0, len(np.unique(df1['block'])))] 
    current_block = (df1['block'][0], 0)
    for i in range(0, len(df1['block'])):
        if df1['block'][i] != current_block[0]:
            blocks[current_block[1]][1] = i
            current_block = (df1['block'][i], current_block[1] + 1)
            blocks[current_block[1]][0] = i
    blocks[-1][1] = i + 1
        
    return([df1, df2, blocks])

def create_df(df1, df2, covs):
    new_df = pd.DataFrame()
    columns1 = [c for c in covs if c in df1.columns]
    columns2 = [c for c in covs if c in df2.columns]
    np_df = np.transpose(np.array([df1[c] for c in columns1] + \
                                  [df2[c] for c in columns2]))
    new_df = pd.DataFrame(np_df, columns = columns1 + columns2)
    return(new_df)
    

def sample(df1, df2, formula, family, N, I, T):
    formula = formula.replace(' ', '')
    covs = formula.split('~')[1].split('+')
    Y = [(covs[c], c+1) for c in range(len(covs)) if covs[c] in df2.columns]
    covs = covs + [formula.split('~')[0]]
    formula = formula.replace('~', ' ~ ').replace('+', ' + ')
    
    df1, df2, blocks = create_blocks(df1, df2)

        
    merged_df = create_df(df1, df2, covs)
    B_dict = {}
    P_dict = {}
    print(merged_df)
    if family.lower() == 'normal':
        for i in range(0, len(blocks)):
            B, P = permute_search_normal(merged_df, [blocks[i][0],blocks[i][1]],\
                                         formula, Y, N, I, T)
            B_dict[str(i)] = B
            P_dict[str(i)] = P
    elif family.lower() == 'logistic':
        for i in range(0, len(blocks)):
            B, P = permute_search_logistic(merged_df, [blocks[i][0],blocks[i][1]],\
                                         formula, Y, N, I, T)
            B_dict[str(i)] = B
            P_dict[str(i)] = P
    elif family.lower() == 'poisson':
        for i in range(0, len(blocks)):
            B, P = permute_search_pois(merged_df, [blocks[i][0],blocks[i][1]],\
                                         formula, Y, N, I, T)
            B_dict[str(i)] = B
            P_dict[str(i)] = P
    return([B_dict, P_dict])
            
    
from perm_sample_norm_06 import *
from perm_sample_binom_05 import *
from perm_sample_poisson_04 import *