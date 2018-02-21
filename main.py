#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:15:33 2018

@author: Edwin
"""

from master import *


df1 = simulate_data_normal(50, [0, 3,1, -2])

real_y = list(df1['y'][0:5])
df1['y'][0:5] = np.random.choice(real_y, 5, replace = False)
B, P = permute_search_normal(df1, [0, 5], 'y ~ x1 + x2 + x3', 10, 20, 20)