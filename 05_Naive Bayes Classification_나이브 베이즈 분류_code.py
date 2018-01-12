# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:02:22 2018

@author: A
"""




import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB



'''
####################
        data load
####################
'''

iris = datasets.load_iris()
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris.data, iris.target)

iris.data.shape
iris.feature_names
iris.target_names


cancer = datasets.load_breast_cancer()
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(cancer.data, cancer.target)

cancer.data.shape 
cancer.feature_names
cancer.target_names


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

mnb_iris = MultinomialNB()
mnb_iris_fit = mnb_iris.fit(iris_X_train, iris_y_train)

mnb_iris_predict = mnb_iris_fit.predict(iris_X_test[0].reshape([-1,4]))
print(mnb_iris_predict)

mnb_iris_score = mnb_iris_fit.score(iris_X_test, iris_y_test)
print("Accuracy of iris_test_set :", mnb_iris_score)



gnb_iris = GaussianNB()
gnb_iris_fit = gnb_iris.fit(iris_X_train, iris_y_train)

gnb_iris_predict = gnb_iris_fit.predict(iris_X_test[0].reshape([-1,4]))
print(gnb_iris_predict)

gnb_iris_score = gnb_iris_fit.score(iris_X_test, iris_y_test)
print("Accuracy of iris_test_set :", gnb_iris_score)





'''
####################
        cancer
####################
'''

mnb_cancer = MultinomialNB()
mnb_cancer_fit = mnb_cancer.fit(cancer_X_train, cancer_y_train)

mnb_cancer_predict = mnb_cancer_fit.predict(cancer_X_test[0].reshape([-1,30]))
print(mnb_cancer_predict)

mnb_cancer_score = mnb_cancer_fit.score(cancer_X_test, cancer_y_test)
print("Accuracy of cancer_test_set :", mnb_cancer_score)



gnb_cancer = GaussianNB()
gnb_cancer_fit = gnb_cancer.fit(cancer_X_train, cancer_y_train)

gnb_cancer_predict = gnb_cancer_fit.predict(cancer_X_test[0].reshape([-1,30]))
print(gnb_cancer_predict)

gnb_cancer_score = gnb_cancer_fit.score(cancer_X_test, cancer_y_test)
print("Accuracy of cancer_test_set :", gnb_cancer_score)





'''
####################
        MNIST
####################
'''

mnb_MNIST = MultinomialNB()
mnb_MNIST_fit = mnb_MNIST.fit(MNIST_X_train, MNIST_y_train)

mnb_MNIST_predict = mnb_MNIST_fit.predict(MNIST_X_test[0].reshape([-1,784]))
print(mnb_MNIST_predict)

mnb_MNIST_score = mnb_MNIST_fit.score(MNIST_X_test, MNIST_y_test)
print("Accuracy of MNIST_test_set :", mnb_MNIST_score)



gnb_MNIST = GaussianNB()
gnb_MNIST_fit = gnb_MNIST.fit(MNIST_X_train, MNIST_y_train)

gnb_MNIST_predict = gnb_MNIST_fit.predict(MNIST_X_test[0].reshape([-1,784]))
print(gnb_MNIST_predict)

gnb_MNIST_score = gnb_MNIST_fit.score(MNIST_X_test, MNIST_y_test)
print("Accuracy of MNIST_test_set :", gnb_MNIST_score)














