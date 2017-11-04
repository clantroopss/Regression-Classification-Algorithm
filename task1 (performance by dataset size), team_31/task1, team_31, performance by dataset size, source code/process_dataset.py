# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 03:45:44 2017

@author: merb
"""
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors
from sklearn import tree

def process_dataset(ds, name, algorithm):
    x = np.array([])
    y = np.array([])

    if name == 'sum_with_noise':
        if algorithm == "regression":
            x,y = get_xy_reg_sum_with_noise(ds)
        else:
            x,y= get_xy_clf_sum_with_noise(ds)
            
    if name == 'sum_without_noise':
        if algorithm == "regression":
            x,y = get_xy_reg_sum_without_noise(ds)
        else:
            x,y = get_xy_clf_sum_without_noise(ds)

    if name == 'skin_nonskin':
        if algorithm == "regression":
            x,y = get_xy_reg_skin_nonskin(ds)
        else:
            x,y = get_xy_clf_skin_nonskin(ds)
           
    if name == '3d_road_network':
        if algorithm == "regression":
            x,y = get_xy_reg_3d_road(ds)
        else:
            x,y = get_xy_clf_3d_road(ds)
    return x, y

def get_xy_reg_sum_with_noise(ds): 
    ds = ds.drop('Noisy Target Class', axis=1)  # dropping class
    x = ds.drop(['Noisy Target'], axis=1)   # creating feature set
    y = ds[['Noisy Target']]                # creating target set
    return x,y

def get_xy_clf_sum_with_noise(ds): 
    ds['Noisy Target Class (Encoded)'] = ds['Noisy Target Class'].astype('category')
    ds['Noisy Target Class_codes'] = ds['Noisy Target Class (Encoded)'].cat.codes
    x = ds.drop(['Noisy Target', 'Noisy Target Class', 'Noisy Target Class (Encoded)', 'Noisy Target Class_codes'],axis=1)
    y = ds[['Noisy Target Class_codes']]
    return x,y

def get_xy_reg_sum_without_noise(ds):
    ds = ds.drop('Target Class', axis=1)    # dropping class
    x = ds.drop('Target', axis=1)           # creating feature set
    print "printing x=",x
    y = ds[['Target']]
    return x,y

def get_xy_clf_sum_without_noise(ds):
    ds['Target Class (Encoded)'] = ds['Target Class'].astype('category')
    ds['Target Class_codes'] = ds['Target Class (Encoded)'].cat.codes
    x = ds.drop(['Target', 'Target Class', 'Target Class (Encoded)', 'Target Class_codes'],axis=1)
    y = ds[['Target Class_codes']]
    return x,y

def get_xy_reg_skin_nonskin(ds):
    x = ds.iloc[:, 0:3]
    y = ds.iloc[:, 3]                     # Target set
    return x,y

def get_xy_clf_skin_nonskin(ds):
    ds.columns = ["R", "G" , "B", "Skin_NonSkin"]
    x = ds.iloc[:, 0:3].values
    y = ds["Skin_NonSkin"]
    return x,y

def get_xy_reg_3d_road(ds):
    x = ds.iloc[:, 1:3]
    y = ds.iloc[:, 3:]
    return x,y

def get_xy_clf_3d_road(ds):
    x = ds.iloc[:, 1:3]                     # creating feature set -- using columns 1 and 2; not using 0th
    # Create labels
    target_column = 3
    y_ = ds[target_column]
    # Labels created for values of each specified quantile (eg. 1/3rd is low, 1/3rd medium, 1/3rd high)
    first = y_.quantile(.33)
    second = y_.quantile(.67)
    y = ds.apply(lambda row: create_label_utility(row[target_column], first, second), axis=1)  # Target set
    return x,y

def create_label_utility(target, first, second):
    if target < first:
        return 0
    elif first <= target < second:
        return 1
    else:
        return 2

