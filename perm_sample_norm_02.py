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

from master import build_permutation, glm_mcmc_inference

from scipy.stats import norm, lognorm, poisson
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


def normal_permute(x, y, p, T, m, sd):

    likelihood = sum([np.log(l) for l in norm.pdf(x, y, sd)])
    print(likelihood)
     
    y_t = list(y)
    for t in range(T): 
        i, j = np.random.choice(m, 2, replace = False)
        new_l = sum([np.log(l) for l in norm.pdf([x[i], x[j]], [y_t[j], y_t[i]], sd)])
        old_l = sum([np.log(l) for l in norm.pdf([x[i], x[j]], [y_t[i], y_t[j]], sd)])
        
        choice = min(1, np.exp(new_l - old_l))
        rand = np.random.rand()
        if rand <= choice:
            temp = y_t[i]
            y_t[i] = y_t[j]
            y_t[j] = temp
            
            temp = p[i]
            p[i] = p[j]
            p[j] = temp            
    #Returns log likelihood        
    return([p, sum([np.log(l) for l in norm.pdf(x, y_t, sd)])])
            
def permute_search_normal(df,formula, N, I, T):
    #N: Number of permutations
    #I: Number of samples in sampling Betas
    #T: Number of iterations in row swapping phase
    
    #Initialize output arrays
    #B: Betas for T samplings
    B = [0 for i in range(N)] 
    #P: Permutations after I iterations for each set of Betas
    P = list(B)
    #L: Log Likelihoods of permutations in P with Betas in B
    L = list(B)
    
    y1 = formula.split(' ~ ')[0]    
    
    A = pd.DataFrame.as_matrix(df.drop(str(y1), 1))
    m, n = len(A[:,0]), len(A[0,:])+1
    A = sp.sparse.coo_matrix(np.concatenate([np.ones((m, 1)), A], 1))
    
    #N iterations
    P[-1] = [i for i in range(m)]
    for t in range(N):
        #Input is the data in the order of the last permutation
        new_y = build_permutation(P[t-1], list(df[y1]))
        df[y1] = new_y   
        
        y = pd.DataFrame.as_matrix(df[str(y1)])
        
        #Sample Betas and search for permutations
        trace = glm_mcmc_inference(df, formula, pm.glm.families.Normal(), I)
        beta_names = ['Intercept']
        beta_names.extend(formula.split(' ~ ')[1].split(' + '))
        b = np.transpose([trace.get_values(s)[-1] for s in beta_names])
        sd = trace.get_values('sd')[-1]
        
        x = A * b
        
        B[t] = b
        P[t], L[t] = normal_permute(x, y, list(P[t-1]), T, m, sd)
        
    return([B, P, L])
    