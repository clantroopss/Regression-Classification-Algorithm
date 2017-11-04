# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 04:25:43 2017

@author: merb
"""
import process_dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

    
def split(model, x, y, algorithm):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    if algorithm == "classification":
        print "Accuracy score:",metrics.accuracy_score(y_test,y_predict)
        print "F1 score:", metrics.f1_score(y_test,y_predict,average="micro")
    elif algorithm == "regression":
        print "Root mean squared error:", np.sqrt(metrics.mean_squared_error(y_test,y_predict))
        print "Mean absolute error:", metrics.mean_absolute_error(y_test, y_predict)

def cross_valid(regression_model, x, y, scoring):
    scores = cross_validate(regression_model, x, np.squeeze(y), cv=10, scoring=scoring)

    for score in scoring:
        if score == "neg_mean_squared_error":
            print "Root mean square error value: ",np.sqrt(-scores['test_' + score].mean())
        elif score == "neg_mean_absolute_error":
            print "Mean Absolute Error value:", (-scores['test_' + score].mean())
        else:
            print score ,"value: ",scores['test_' + score].mean()

def execute_algorithm(ds, name, chunk_size, algorithm):
    if algorithm == "classification":
        x, y = process_dataset.process_dataset(ds, name, "classification")
        scoring = ['accuracy', 'f1_macro']
        print "Logistic Regression"
        cross_valid(LogisticRegression(), x, y, scoring)       # Logistic Regression
        print "Decision tree classifier"
        regression_model = DecisionTreeClassifier() 
        
    if algorithm == "regression":
        x, y = process_dataset.process_dataset(ds, name, "regression")
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
        print "Linear Regression"
        split(LinearRegression(), x, y, algorithm)    # Linear Regression
        print "KNN"
        regression_model = KNeighborsRegressor(n_neighbors=2)  # SGDRegressor

    assert x.size != 0 and y.size != 0
    if chunk_size < 50000:
        cross_valid(regression_model, x, y, scoring)
    else:
        split(regression_model, x, y, algorithm)
    print "\n"





        

        

