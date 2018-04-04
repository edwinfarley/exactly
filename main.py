#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:15:33 2018

@author: Edwin
"""

from master import *

if __name__ == '__main__':
    df1 = simulate_data_normal(20, [0, 3,1, -2])
    
    real_y = list(df1['y'][2:6])
    df1['y'][0:4] = np.random.choice(real_y, 4, replace = False)
    real_x = list(df1['x3'][2:6])
    df1['y'][0:4] = np.random.choice(real_x, 4, replace = False)
    df1['y'][2] = None
    df1['y'][18] = None
    df1['block'] = [0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,4,4,5,5,5]
    print(df1)
    df2 = pd.DataFrame(df1['y'])
    df2['y1'] = pd.DataFrame(df1['x3'])
    df2['block'] = [0,0,1,1,1,1,2,2,2,2,2,2,3,3,3,4,4,5,5,5]
    df1 = df1.drop('y',1)
    B, P = sample(df1, df2, 'y ~ x1 + x2 + y1', 'Normal', 3, 20, 20)