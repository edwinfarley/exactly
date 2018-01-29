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

    df = pd.DataFrame(
        {str("x1"): np.random.RandomState(seed).normal(
                0, v, N)})
    for i in range(2, len(B)):
        print(i)
        df_i = pd.DataFrame(
                {str("x" + str(i)): np.random.RandomState(seed+i).normal(
                0, v, N)})
        df = pd.concat([df, df_i], axis = 1)


    # Use a linear model (y ~ beta_0 + beta_1*x + epsilon) to
    # generate a column 'y' of responses based on 'x'
    #Betas are normally distributed with mean 0 and variance eps_sigma_sq
    y = B[0] + np.matmul(pd.DataFrame.as_matrix(df), np.transpose(B[1:])) + np.random.RandomState(42).normal(0, eps_sigma_sq, N)
    df["y"] =  y

    return df

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
    new = [0 for i in range(len(arr))] 
    l = len(p)
    for i in range(l):
            new[i] = arr[p[i]]
    return(new)

def normal_permute(df,formula, I, T):
    family = 'Normal'
    y1 = formula.split(' ~ ')[0]
    print(y1)
    
    y = pd.DataFrame.as_matrix(df[str(y1)])
    A = pd.DataFrame.as_matrix(df.drop(str(y1), 1))
    m, n = len(A[:,0]), len(A[0,:])+1
    A = np.concatenate([np.ones((m, 1)), A], 1)
    
    #P = eye(m)
    p = [i for i in range(m)]
    
    if family == 'Normal':
            with pm.Model() as P_model:
                pm.glm.GLM.from_formula(formula, df, family = pm.glm.families.Normal())
                trace = pm.sample(I, chains = 1, tune = 0, progress_bar = False)
                print("Sampling done.")
                
            #trace = glm_mcmc_inference(df, formula, pm.glm.families.Normal(), I)
            beta_names = ['Intercept']
            beta_names.extend(formula.split(' ~ ')[1].split(' + '))
            B = np.transpose([trace.get_values(s)[-1] for s in beta_names])
            sd = trace.get_values('sd')[-1]
    
            x = np.matmul(A, B)
    
            likelihood = sum([np.log(l) for l in norm.pdf(x, y, sd)])
            print(likelihood)
    
    elif family == 'Poisson':
        trace = glm_mcmc_inference(df, formula, pm.glm.families.Poisson(), I)
        beta_names = ['Intercept']
        beta_names.extend(formula.split(' ~ ')[1].split(' + '))
        B = np.transpose([trace.get_values(s)[-1] for s in beta_names])
        mu = trace.get_values('mu')[-1]
        
        x = np.matmul(A, B)
        
        likelihood = np.sum([np.log(l) for l in poisson.pdf(x, mu)])
        print(likelihood)
     
    y_t = list(y)
    for t in range(T): 
        '''print(p)
        for i in range(m):
            df[y1][i] = y[p[i]] '''
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
    return([B, p, sum([np.log(l) for l in norm.pdf(x, y_t, sd)])])
            
def permute_search_normal(df,formula, I, T):
    df_x = pd.DataFrame(df)
    y1 = formula.split(' ~ ')[0]
    print(y1)
    
    B = [0 for i in range(T)] 
    P = list(B)
    L = list(B)
    for t in range(T):
        if t > 0:
            new_y = build_permutation(P[t-1], list(df_x[y1]))
            df_x[y1] = new_y
        B[t], P[t], L[t] = normal_permute(df_x,formula, 'Normal', 2000, I)
        
    return([B, P, L])
    
'''  
df1 = simulate_data_normal(40, [2, -1, 3])
real_y = list(df1['y'])
df1['y'] = np.random.choice(real_y, 40, replace = False)
B, P, L = permute_search_normal(df1, 'y ~ x1 + x2 + x3', 1000, 10)
'''

