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
import scipy as sp
import math

from master import build_permutation, glm_mcmc_inference

from scipy.stats import norm, lognorm, poisson, binom
#from localsearch import *

eps_sigma_sq = 1
v=1


def simulate_data_logistic(N, B):
    """
    Simulate a random dataset using a noisy
    linear process.

    N: Number of data points to simulate
    B: Vector of regression parameters. Defines number of covariates. (Include B_0)
    """
    seed = 7

    df = pd.DataFrame(
        {str("x1"): np.random.RandomState(seed).uniform(size = N)})
    for i in range(2, len(B)):
        print(i)
        df_i = pd.DataFrame(
                {str("x" + str(i)): np.random.RandomState(seed+i).uniform(size = N)})
        df = pd.concat([df, df_i], axis = 1)


    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of responses based on 'x'
    #Betas are normally distributed with mean 0 and variance eps_sigma_sq
    p = np.exp(B[0] + np.matmul(pd.DataFrame.as_matrix(df), np.transpose(B[1:])))
    p = p/(1+p)#+ np.random.RandomState(42).normal(0, eps_sigma_sq, N)
    df["y"] = np.round(p)

    return df


    
def poisson_permute(x, y, p, T, m):
    #Wasserman pg. 223
    likelihood = sum([y[i]*x[i] - np.exp(y[i]*x[i]) - np.log(math.factorial(y[i])) for i in range(0, len(p))])
    print(likelihood)
     
    y_t = list(y)
    for t in range(T): 
        i, j = np.random.choice(m, 2, replace = False)
        new_l = y_t[i]*x[i] - np.exp(y_t[i]*x[i]) - np.log(math.factorial(y_t[i])) + y_t[j]*x[j] - np.exp(y_t[j]*x[j]) - np.log(math.factorial(y_t[j]))
        old_l = y_t[j]*x[i] - np.exp(y_t[j]*x[i]) - np.log(math.factorial(y_t[j])) + y_t[i]*x[j] - np.exp(y_t[i]*x[j]) - np.log(math.factorial(y_t[i]))
        
        choice = min(1, np.exp(new_l - old_l))
        rand = np.random.rand()
        if rand <= choice:
            temp = y_t[i]
            y_t[i] = y_t[j]
            y_t[j] = temp
            
            temp = p[i]
            p[i] = p[j]
            p[j] = temp
             
    return([p, sum([y[i]*x[i] - np.exp(y[i]*x[i]) - np.log(math.factorial(y[i])) for i in range(0, len(p))])])

#np.prod([P[i]**y_t[i]*(1-P[i])**(1-y_t[i]) for i in range(0, len(P))])
            
def permute_search_pois(df,formula, N, I, T):
    ##N: Number of permutations
    #I: Number of samples in sampling Betas
    #T: Number of iterations in row swapping phase

    #Initialize output arrays
    #P: Permutations after I iterations for each set of Betas
    #L: Log Likelihoods of permutations in P with Betas in B

    y1 = formula.split(' ~ ')[0]
    covariates = formula.split(' ~ ')[1].split(' + ')
    block_df = pd.DataFrame()
    for col in covariates:
        block_df.loc[:, col] = df[col][block[0]:block[1]]
    block_df.loc[:, y1] = df[y1][block[0]:block[1]]
    A = pd.DataFrame.as_matrix(block_df.drop(str(y1), 1))
    m, n = len(A[:,0]), len(A[0,:])+1
    A = sp.sparse.coo_matrix(np.concatenate([np.ones((m, 1)), A], 1))

    #N iterations

    #P: Permutations after I iterations for each set of Betas
    P = list(range(block[0], block[1]))*N
    #B: Betas for T samplings
    B = [0 for i in range(n)]*N
    for t in range(N):
        #Input is the data in the order of the last permutation
        if t == 0:
            new_y = build_permutation(P[0:m], list(block_df[y1]))
        else:
            new_y = build_permutation(P[((t-1)*m):(t*m)], list(block_df[y1]))

        df[y1][block[0]:block[1]] = new_y

        y = pd.DataFrame.as_matrix(block_df[str(y1)])
        trace = glm_mcmc_inference(df, formula, pm.glm.families.Poisson(), I)
        beta_names = ['Intercept']
        beta_names.extend(formula.split(' ~ ')[1].split(' + '))
        b = np.transpose([trace.get_values(s)[-1] for s in beta_names])
        
        x = A * b
        
        B[((t)*n):((t+1)*n)] = b
        if t == 0:
            P[0:m] = normal_permute(x, y, list(P[0:m]), T, m, sd)
        else:
            P[((t)*m):((t+1)*m)] = poisson_permute(x, y, list(P[((t-1)*m):(t*m)]), T, m)

    return([B, P])