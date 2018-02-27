#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 18:56:04 2018

@author: Edwin
"""

import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import math


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
    
#from perm_sample_norm_03 import *
from perm_sample_norm_r import *
from perm_sample_binom_r import *
from perm_sample_poisson_r import *
    

    
