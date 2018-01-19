# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:14:57 2018

@author: admin
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree


Example = np.array([[47.4, 16.4, '0'],
                    [85.5, 16.8, '1'],
                    [64.8, 17.2, '0'],
                    [66, 18.4, '0'],
                    [64.8, 21.6, '1'],
                    [51, 22, '1']])





from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Example[:,:2],Example[:,2])


import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=["Income","Lotsize"],  
                         class_names=['0','1'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph



