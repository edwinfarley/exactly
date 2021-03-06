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
import random

from master import build_permutation, glm_mcmc_inference

from scipy.stats import norm
#from localsearch import *

eps_sigma_sq = 1
v=1


def simulate_data_normal(N, B):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    B: Vector of regression parameters. Defines number of covariates. (Include B_0)
    """
    # Create a pandas DataFrame with column 'x' containing
    # N uniformly sampled values between 0.0 and 1.0
    seed = 7
    print(1)
    df = pd.DataFrame(
        {str("x1"): np.random.RandomState().normal(
                0, v, N)})
    for i in range(2, len(B)):
        print(i)
        df_i = pd.DataFrame(
                {str("x" + str(i)): np.random.RandomState().normal(
                0, v, N)})
        df = pd.concat([df, df_i], axis = 1)


    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of responses based on 'x'
    #Betas are normally distributed with mean 0 and variance eps_sigma_sq
    y = B[0] + np.matmul(pd.DataFrame.as_matrix(df), np.transpose(B[1:])) + np.random.RandomState(42).normal(0, eps_sigma_sq, N)
    df["y"] =  y

    return df


def normal_permute(A, b, y, p, T, m, sd, Y):

    #likelihood = sum([np.log(l) for l in norm.pdf(x, y, sd)])
    #print(likelihood)

    for t in range(T):
        i, j = np.random.choice(m, 2, replace = False)
        
        x_i = np.matmul(A[i, :], b)
        x_j = np.matmul(A[j, :], b)
        
        #Switch relevant (Y) covariates
        for _, ix in Y:
                temp = A[i, ix]
                A[i, ix] = A[j, ix]
                A[j, ix] = temp
        x_i_swap = np.matmul(A[i, :], b)
        x_j_swap = np.matmul(A[j, :], b)
        
        new_l = sum([np.log(l) for l in norm.pdf([x_i_swap, x_j_swap], [y[j], y[i]], sd)\
                     *((np.isnan([y[j], y[i]]) * -2) +1)])
        old_l = sum([np.log(l) for l in norm.pdf([x_i, x_j], [y[i], y[j]], sd)\
                     *((np.isnan([y[i], y[j]]) * -2) +1)])

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
            for _, ix in Y:
                temp = A[i, ix]
                A[i, ix] = A[j, ix]
                A[j, ix] = temp
    #Returns permutation, y
    return(p, y)

def permute_search_normal(df, block, formula, Y, N, I, T, burnin, interval):
    #N: Number of permutations
    #I: Number of samples in sampling Betas
    #T: Number of iterations in row swapping phase

    #Initialize output arrays
    #P: Permutations after I iterations for each set of Betas
    #L: Log Likelihoods of permutations in P with Betas in B
    y1 = formula.split(' ~ ')[0]
    covariates = formula.split(' ~ ')[1].split(' + ')
    num_X = len(covariates) - len(Y)
    
    #Y = Y + [(y1, -1)]
    #missing = np.arange(len(df[y1]))
    #missing = missing[np.where(np.isnan(df[y1]))]
    #missing = missing[missing < block[1]]
    #missing = missing[missing >= block[0]]
    
    #df[y1].loc[np.isnan(df[y1])] = np.random.uniform(min(df[y1]), max(df[y1]))
    #block_df = pd.DataFrame()
    #for col in covariates:
    #    block_df.loc[:, col] = df[col][block[0]:block[1]]
    block_df = pd.DataFrame(df[block[0]:block[1]]).reset_index(drop=True)
    #block_df.loc[:, y1] = df[y1][block[0]:block[1]]
    #print(df[y1])
    
    #A = pd.DataFrame.as_matrix(block_df)
    #m, n = len(A[:,0]), len(A[0,:])+1
    #A = np.concatenate([np.ones((m, 1)), A], 1)
    X_missing = np.where(np.isnan(block_df[covariates[0]]))[0]
    num_X_missing = len(X_missing)
    num_finite = len(block_df) - num_X_missing
    num_Y_missing = sum(np.isnan(block_df[y1]))
    #print(X_missing)
    m, n = len(block_df), len(block_df.columns)+1

    
    #Remove NaNs outside of current block
    df = pd.concat([df[0:block[0]].dropna(), df[block[0]:block[1]], df[block[1]:].dropna()])
    #N iterations
    #sum(block_df[y1].notnull())
    block_size = sum(block_df[covariates[0]].notnull())
    print(block_size)
    #P: Permutations after I iterations for each set of Betas
    P = np.zeros((N, block_size)).astype(int)
    #B: Betas for T samplings
    B = [0 for i in range(n)]*N
    
    original_block = pd.DataFrame(block_df)
    for i in X_missing:
                r = int(num_finite * random.random())
                #print((block_df.sort_values(by = y1).drop(y1, 1))[r:r+1])
                #print(block_df.loc[i, :][:num_X])
                block_df.loc[i, :][:num_X] = list((block_df.sort_values(by = y1).drop(y1, 1)).loc[r,:][:num_X])
    
    print(block_df)
    for t in range(burnin + (N*interval)):
        #block_df.loc[:, y1] = df[y1].loc[block[0]:block[1]-1]
        #Input is the data in the order of the last permutation
        if t > 0:
            df[y1].loc[block[0]:block[1]-1] = new_y
            block_df[y1] = new_y
            for col, _ in Y:
                new_col = build_permutation(P_t, list(original_block[col]))
                df[col].loc[block[0]:block[1]-1] = new_col
                block_df[col] = new_col   
            if num_X_missing:
                block_df['y_b'] = np.matmul(A, b)
                print(block_df)
                for i in X_missing:
                    r = int(num_finite * random.random())
                    block_df.loc[i, :][:num_X] = list((block_df.sort_values(by = 'y_b').drop([y1, 'y_b'], 1)).loc[r,:][:num_X])
                block_df = block_df.drop('y_b', 1)
                print(block_df)
        #y = pd.DataFrame.as_matrix(block_df[y1])
        
        #Sample Betas and search for permutations
        trace = glm_mcmc_inference(df, formula, pm.glm.families.Normal(), I)
        beta_names = ['Intercept']
        beta_names.extend(formula.split(' ~ ')[1].split(' + '))
        b = np.transpose([trace.get_values(s)[-1] for s in beta_names])
        sd = trace.get_values('sd')[-1]
        
        A = pd.DataFrame.as_matrix(block_df.drop(y1, 1))
        A = np.concatenate([np.ones((m, 1)), A], 1)
        print(A)
        B[((t)*n):((t+1)*n)] = b
        
        
        if t == 0:
            P_t, new_y = normal_permute(A, b, np.array(block_df[y1]), np.arange(0, m), T, m, sd, Y)
            #P_t[np.where(np.isfinite(new_y))]
            #P[0, :] = [p for i, p in enumerate(P_t) if i not in X_missing][np.where(np.isfinite(new_y))[0][:-X_missing]]
            if burnin == 0:
                if num_X_missing:
                    P[0, :] = P_t[:-num_X_missing]
                elif num_Y_missing:
                    #[np.where(np.isfinite(new_y))]
                    temp = np.array(P_t)
                    temp[np.where(np.isnan(new_y))] = -(block[0]+1)
                    P[0, :] = temp
                else:
                    P[0, :] = P_t   
        else:
            #list(P[((t-1)*block_size):(t*block_size)])
            P_t, new_y = normal_permute(A, b, np.array(new_y), P_t, T, m, sd, Y)
            #P[((t)*block_size):((t+1)*block_size)]
            #P[t, :] = P_t[np.where(np.isfinite(new_y))]
            if t >= burnin:
                if (t-burnin)%interval == 0:
                    if num_X_missing:
                        # P_t[np.where(np.isfinite(new_y))[:-num_missing]]
                        P[int((t-burnin)/interval), :] = P_t[:-num_X_missing]
                    elif num_Y_missing:
                        #[np.where(np.isfinite(new_y))]
                        temp = np.array(P_t)
                        temp[np.where(np.isnan(new_y))] = -(block[0]+1)
                        P[int((t-burnin)/interval), :] = temp
                    else:
                        P[int((t-burnin)/interval), :] = P_t 
        #print(df[y1][block[0]:block[1]])
        #print(new_y)

    return([B, P])
