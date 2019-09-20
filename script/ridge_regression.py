# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from math import sqrt
import os
import matplotlib.pyplot as plt

def ridgeReg(X, y, lamda):
    one = [[1 for i in range(X.shape[1])]]
    X = np.append(X, one, axis = 0)
    Xt = np.transpose(X)
    I = np.identity(X.shape[0] - 1)
    I = np.identity(X.shape[0])
    zeros = np.zeros((X.shape[0] - 1, 1))
    I = np.append(I, [zeros], axis = 1)
    # another list of zeros
    zeros = [[0 for i in range(I.shape[1])]]
    I = np.append(I, zeros, axis = 0)
    C1 = np.matmul(X, Xt)
    C2 = lamda * I
    C = C1 + C2
    Cinv = np.linalg.inv(C)
    d = np.matmul(X, y)
    
    w = np.matmul(Cinv, d)
    # getting bias term from the weights
    bias = w[-1]
    w = w[:-1]
    
    # now calculating the objective function
    wtsqr = w*w
    obj1 = lamda * np.sum(wtsqr)
    wtrp = np.transpose(w)
    X = X[:-1, :]
    term1 = np.matmul(wtrp, X)
    term2 = term1 + bias
    term3 = term2 - y
    obj2 = np.sum(term3 * term3)
    obj = obj1 + obj2
    list = [w, bias, obj]
    return list
    
    

# a function to perform the LOOCV
def loocv(X, y, lamda):
    sum_squared_error = 0
    list = ridgeReg(X, y, lamda)
    for i in range(X.shape[1]):
        xi = X[:, i]
        yi = y[i]
        #X = np.delete(X, i, 1)
        #y = np.delete(y, i, 0)
        #params = ridgeReg(X, y, lamda)
        w = list[0]
        wtrp = np.transpose(w)
        predicted = np.matmul(wtrp, xi)
        predicted = predicted[0]
        error = predicted - yi
        esquared = error * error
        sum_squared_error = sum_squared_error + esquared
    
    list.append(sum_squared_error)
    return list


os.chdir('C:\\Users\\admin\\Desktop\\PostG\\GRE\\Second Take\\Applications\\Univs\\Stony Brook\\Fall 19 Courses\\ML\\Homeworks\\Homework 2\\Kaggle\\Data')
train_data = pd.read_csv('trainData.csv', header = None)
train_data = train_data.iloc[:, 1:]
# convert into numpy and take transpose to send into the ridge function
np_train_dat = train_data.values
np_train_dat = np.transpose(np_train_dat)

# read the training labels too
train_label = pd.read_csv('trainLabels.csv', header = None)
train_label = train_label.iloc[:, 1:]
np_train_label = train_label.values

# similarly, reading the validation data and labels
val_data = pd.read_csv('valData.csv', header = None)
val_data = val_data.iloc[:, 1:]
np_val_dat = val_data.values
np_val_dat = np.transpose(np_val_dat)

val_label = pd.read_csv('valLabels.csv', header = None)
val_label = val_label.iloc[:, 1:]
np_val_label = val_label.values


# Modeling the training data now
#params = loocv(np_train_dat, np_train_label, 1)
datum = np_train_dat.T
thresholder = VarianceThreshold(threshold = 0.0001)
ab = thresholder.fit_transform(datum)
ab = np.log(ab + 1)
np_train_dat = ab.T
params = ridgeReg(np_train_dat, np_train_label, 0.01)
wt = params
prediction = np.matmul(np.transpose(np_train_dat), wt)
rms = sqrt(mean_squared_error(np_train_label, prediction))



# Modeling the training + validation data
train = np.append(np_train_dat, np_val_dat, axis = 1)
train = np.append(train, np_test_dat.T, axis = 1)
target = np.append(np_train_label, np_val_label, axis = 0)


# removing low variance
traint = train.T
thresholder = VarianceThreshold(threshold = 0.0003)
ab = thresholder.fit_transform(traint)
train = ab[:10000, :].T
np_test_dat = ab[10000:, :]
params = ridgeReg(train, target, 0.01)




# plotting correlation plot
dataframe = np.append(train.T, target, axis = 1)
dataframe = pd.DataFrame(data = dataframe)
subset = dataframe.iloc[:, 1:20]
subset.insert(19, '20', value = dataframe.iloc[:, -1])
corr = subset.corr()
corr.style.background_gradient(cmap = 'coolwarm')
plt.matshow(subset.corr())
plt.show()









# Predicting on the test data
test_data = pd.read_csv('testData_new.csv', header = None)
test_data = test_data.iloc[:, 1:]
np_test_dat = test_data.values
wt = params
prediction = np.matmul(np_test_dat, wt)

prediction[prediction < 80] = 80
prediction[prediction > 100] = 100

# creating my first submission
id_no = np.array([i for i in range(prediction.shape[0])]).T
id_no = np.reshape(id_no, (4749, 1))
submission = np.append(id_no, prediction, axis = 1).astype(int)
df = pd.DataFrame(data = submission, columns = ['Id', 'Expected'])
df.to_csv('submission.csv', index = False)

