#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:52:11 2017

@author: Edwin
"""

'''
With a permutation, sample B.
Now sample permutations using metropolis method. End on some permutation: this permutation goes with that B.
Start over and repeat a set number of times.
'''


import numpy as np
import pandas as pd
import pymc3 as pm
import math

from master import build_permutation, glm_mcmc_inference

#from localsearch import *

def poisson_permute(A, b, y, p, T, m, Y):
    #likelihood = sum([y[i]*x[i] - np.exp(y[i]*x[i]) - np.log(math.factorial(y[i])) for i in range(0, len(p))])
    #print(likelihood)
     
    for t in range(T): 
        i, j = np.random.choice(m, 2, replace = False)
        
        x_i = np.matmul(A[i, :], b)
        x_j = np.matmul(A[j, :], b)
        
        #Switch relevant (Y) covariates
        for _, ix in Y[:-1]:
                temp = A[i, ix]
                A[i, ix] = A[j, ix]
                A[j, ix] = temp
        x_i_swap = np.matmul(A[i, :], b)
        x_j_swap = np.matmul(A[j, :], b)
        
        new_l = (y[j]*x_i_swap) - np.exp(y[j]*x_i_swap) - np.log(math.factorial(y[j])) + (y[i]*x_j_swap) - np.exp(y[i]*x_j_swap) - np.log(math.factorial(y[i]))
        old_l = (y[i]*x_i) - np.exp(y[i]*x_i) - np.log(math.factorial(y[i])) + (y[j]*x_j) - np.exp(y[j]*x_j) - np.log(math.factorial(y[j]))
        
        choice = min(1, np.exp(new_l - old_l))
        rand = np.random.rand()
        if rand <= choice:
            temp = y[i]
            y[i] = y[j]
            y[j] = temp
            
            temp = p[i]
            p[i] = p[j]
            p[j] = temp
        else:
            #Switch Y covariates back
            for _, ix in Y[:-1]:
                temp = A[i, ix]
                A[i, ix] = A[j, ix]
                A[j, ix] = temp
             
    return(p, y)
    
#sum([y[i]*x[i] - np.exp(y[i]*x[i]) - np.log(math.factorial(y[i])) for i in range(0, len(p))])
#np.prod([P[i]**y_t[i]*(1-P[i])**(1-y_t[i]) for i in range(0, len(P))])
            
def permute_search_pois(df, block, formula, Y, N, I, T):
    #N: Number of permutations
    #I: Number of samples in sampling Betas
    #T: Number of iterations in row swapping phase

    #Initialize output arrays
    #P: Permutations after I iterations for each set of Betas
    #L: Log Likelihoods of permutations in P with Betas in B

    y1 = formula.split(' ~ ')[0]
    covariates = formula.split(' ~ ')[1].split(' + ')
    num_X = len(covariates) - len(Y)
    
    block_df = pd.DataFrame(df[block[0]:block[1]]).reset_index(drop=True)

    X_missing = np.where(np.isnan(block_df[covariates[0]]))[0]
    num_missing = len(X_missing)
    num_finite = len(block_df) - num_missing
    print(X_missing)
    m, n = len(block_df), len(block_df.columns)+1

    #Remove NaNs outside of current block
    df = pd.concat([df[0:block[0]].dropna(), df[block[0]:block[1]], df[block[1]:].dropna()])
    #N iterations
    block_size = min(sum(block_df[y1].notnull()), sum(block_df[covariates[0]].notnull()))
    print(block_size)
    #P: Permutations after I iterations for each set of Betas
    P = np.zeros((N, block_size)).astype(int)
    #B: Betas for T samplings
    B = [0 for i in range(n)]*N
    
    original_block = pd.DataFrame(block_df)
    for i in X_missing:
                r = int(num_finite * random.random())
                print((block_df.sort_values(by = y1).drop(y1, 1))[r:r+1])
                print(block_df.loc[i, :][:num_X])
                block_df.loc[i, :][:num_X] = list((block_df.sort_values(by = y1).drop(y1, 1)).loc[r,:][:num_X])

    for t in range(N):
        #Input is the data in the order of the last permutation
        if t > 0:
            df[y1].loc[block[0]:block[1]-1] = new_y
            block_df[y1] = new_y
            for col, _ in Y:
                new_col = build_permutation(P_t, list(original_block[col]))
                df[col].loc[block[0]:block[1]-1] = new_col
                block_df[col] = new_col    
            if num_missing:
                block_df['y_b'] = np.matmul(A, b)
                print(block_df)
                for i in X_missing:
                    r = int(num_finite * random.random())
                    block_df.loc[i, :][:num_X] = list((block_df.sort_values(by = 'y_b').drop([y1, 'y_b'], 1)).loc[r,:][:num_X])
                block_df = block_df.drop('y_b', 1)

        #Sample Betas and search for permutations
        trace = glm_mcmc_inference(df, formula, pm.glm.families.Poisson(), I)
        beta_names = ['Intercept']
        beta_names.extend(formula.split(' ~ ')[1].split(' + '))
        b = np.transpose([trace.get_values(s)[-1] for s in beta_names])

        A = pd.DataFrame.as_matrix(block_df.drop(y1, 1))
        A = np.concatenate([np.ones((m, 1)), A], 1)
        
        B[((t)*n):((t+1)*n)] = b
        if t == 0:
            P_t, new_y = poisson_permute(A, b, np.array(block_df[y1]), np.arange(0, m), T, m, Y)
            if num_missing:
                P[0, :] = P_t[:-num_missing]
            else:
                P[0, :] = P_t[np.where(np.isfinite(new_y))]
        else:
            P_t, new_y = poisson_permute(A, b, np.array(new_y), P_t, T, m, Y)
            if num_missing:
                P[t, :] = P_t[:-num_missing]
            else:
                P[t, :] = P_t[np.where(np.isfinite(new_y))]
    return([B, P])