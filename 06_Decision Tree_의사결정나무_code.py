# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:26:37 2018

@author: A
"""




import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz




'''
####################
        data load
####################
'''

iris = datasets.load_iris()
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris.data, iris.target)

cancer = datasets.load_breast_cancer()
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(cancer.data, cancer.target)

# http://yann.lecun.com/exdb/mnist/
tmp = open('./MNIST/train-images.idx3-ubyte')
MNIST_X_train = np.fromfile(file=tmp, dtype=np.uint8)
MNIST_X_train = MNIST_X_train[16:].reshape([60000, 784]).astype(np.float)
tmp = open('./MNIST/train-labels.idx1-ubyte')
MNIST_y_train = np.fromfile(file=tmp, dtype=np.uint8)
MNIST_y_train = MNIST_y_train[8:].reshape([60000]).astype(np.float)
tmp = open('./MNIST/t10k-images.idx3-ubyte')
MNIST_X_test = np.fromfile(file=tmp, dtype=np.uint8)
MNIST_X_test = MNIST_X_test[16:].reshape([10000, 784]).astype(np.float)
tmp = open('./MNIST/t10k-labels.idx1-ubyte')
MNIST_y_test = np.fromfile(file=tmp, dtype=np.uint8)
MNIST_y_test = MNIST_y_test[8:].reshape([10000]).astype(np.float)

plt.figure(figsize=(8,8))
plt.imshow(MNIST_X_train[0].reshape([28,28]), cmap='gray')


'''
####################
        iris
####################
'''

DT_iris = DecisionTreeClassifier()
DT_iris_fit = DT_iris.fit(iris_X_train, iris_y_train)

DT_iris_predict = DT_iris.predict(iris_X_test[0].reshape([-1,4]))
print(DT_iris_predict)
print(iris_y_test[0])

DT_iris_score = DT_iris.score(iris_X_test, iris_y_test)
print("Accuracy of iris_test_set :", DT_iris_score)


graph_iris = export_graphviz(DT_iris_fit, 
                             out_file=None, 
                             feature_names=iris.feature_names, 
                             class_names=iris.target_names, 
                             filled=True, 
                             rounded=True, 
                             special_characters=True)  
graph_i = graphviz.Source(graph_iris)  
graph_i




'''
####################
        cancer
####################
'''

DT_cancer = DecisionTreeClassifier()
DT_cancer_fit = DT_cancer.fit(cancer_X_train, cancer_y_train)

DT_cancer_predict = DT_cancer.predict(cancer_X_test[0].reshape([-1,30]))
print(DT_cancer_predict)
print(cancer_y_test[0])

DT_cancer_score = DT_cancer.score(cancer_X_test, cancer_y_test)
print("Accuracy of cancer_test_set :", DT_cancer_score)


graph_cancer = export_graphviz(DT_cancer_fit, 
                               out_file=None, 
                               feature_names=cancer.feature_names, 
                               class_names=cancer.target_names, 
                               filled=True, 
                               rounded=True, 
                               special_characters=True)  
graph_c = graphviz.Source(graph_cancer)  
graph_c




'''
####################
        MNIST (PRUNING X)
####################
'''

DT1_MNIST = DecisionTreeClassifier()
DT1_MNIST_fit = DT1_MNIST.fit(MNIST_X_train, MNIST_y_train)

DT1_MNIST_predict = DT1_MNIST.predict(MNIST_X_test[0].reshape([-1,784]))
print(DT1_MNIST_predict)
print(MNIST_y_test[0])

DT1_MNIST_score = DT1_MNIST.score(MNIST_X_test, MNIST_y_test)
print("Accuracy of MNIST_test_set :", DT1_MNIST_score)


graph1_MNIST = export_graphviz(DT1_MNIST_fit, 
                               out_file=None, 
                               feature_names=[str(i+1)+'th feature' for i in range(4)], 
                               class_names=[str(i) for i in range(3)], 
                               filled=True, 
                               rounded=True, 
                               special_characters=True)  
graph1_M = graphviz.Source(graph1_MNIST)  
graph1_M




'''
####################
        MNIST (PRUNING O - pre pruning : max_depth)
####################
'''

DT2_MNIST = DecisionTreeClassifier()
DT2_MNIST_fit = DT2_MNIST.fit(MNIST_X_train, MNIST_y_train)

DT2_MNIST_predict = DT2_MNIST.predict(MNIST_X_test[0].reshape([-1,784]))
print(DT2_MNIST_predict)
print(MNIST_y_test[0])

DT2_MNIST_score = DT2_MNIST.score(MNIST_X_test, MNIST_y_test)
print("Accuracy of MNIST_test_set :", DT2_MNIST_score)


graph2_MNIST = export_graphviz(DT2_MNIST_fit, 
                               out_file=None, 
                               feature_names=[str(i+1)+'th feature' for i in range(4)], 
                               class_names=[str(i) for i in range(3)], 
                               filled=True, 
                               rounded=True, 
                               special_characters=True)  
graph2_M = graphviz.Source(graph2_MNIST)  
graph2_M




