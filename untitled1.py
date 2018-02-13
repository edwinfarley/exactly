#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:37:42 2018

@author: Edwin
"""

def drop(df):
    y = df.drop('y', 1)
    return(y['x1'])