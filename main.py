#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:15:33 2018

@author: Edwin
"""

from master import *


df1 = simulate_data_normal(5, [1, 3, 1, 2, -1, -2])

real_y = list(df1['y'])
df1['y'] = np.random.choice(real_y, 5, replace = False)
B, P, L = permute_search_normal(df1, 'y ~ x1 + x2 + x3 + x4 + x5', 15, 2000, 1000)