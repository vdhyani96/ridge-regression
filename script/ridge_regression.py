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


# function to perform ridge regression
def perform_ridgeReg(X, y, lamda):
    one = [[1 for i in range(X.shape[1])]]
    X = np.append(X, one, axis = 0)
    Xt = np.transpose(X)
    I = np.identity(X.shape[0] - 1)
    zeros = np.zeros((X.shape[0] - 1, 1))
    I = np.append(I, zeros, axis = 1)
    # another list of zeros
    zeros = [[0 for i in range(I.shape[1])]]
    I = np.append(I, zeros, axis = 0)
    C1 = np.matmul(X, Xt)
    C2 = lamda * I
    C = C1 + C2
    Cinv = np.linalg.inv(C)
    d = np.matmul(X, y)
    
    w = np.matmul(Cinv, d)
    
    # pulling out the bias parameter
    bias = w[-1]
    wt = w[:-1]
    
    # now calculating the objective function
    wtsqr = w*w
    obj1 = lamda * np.sum(wtsqr)
    wtrp = np.transpose(wt)
    X = X[:-1, :]
    term1 = np.matmul(wtrp, X)
    term2 = term1 + bias
    term3 = term2 - y
    obj2 = np.sum(term3 * term3)
    obj = obj1 + obj2
    list = [w, bias, obj, Cinv]
    return list
    
    

# a function to perform the LOOCV
def ridgeReg(X, y, lamda):
    loocv_error = []
    params = perform_ridgeReg(X, y, lamda)
    w = params[0]
    bias = params[1]
    Cinv = params[3]
    ones = [[1 for i in range(X.shape[1])]]
    X = np.append(X, ones, axis = 0)
    for i in range(200):
        xi = X[:, i]
        yi = y[i]
        expn1 = np.matmul(w.T, xi) - yi
        expn2 = np.matmul(xi.T, Cinv)
        expn3 = 1 - np.matmul(expn2, xi)
        error = expn1/expn3
        loocv_error.append(error[0])
        
    w = w[:-1]
    list = [w, bias, params[2], loocv_error]
    return list


# a function to compute RMSE values
def calc_rmse(pred, actual):
    error = pred - actual
    sqr_error = error * error
    mean_sqr_error = np.sum(sqr_error)/pred.shape[0]
    return sqrt(mean_sqr_error)

#params = ridgeReg(np_train_dat, np_train_label, 0.01)
# Reading the data
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


# Modeling the training data now for different lamda
rmse_train = []
rmse_val = []
rmse_loocv = []
for lamda in [0.01, 0.1, 1, 10, 100, 1000]:
    params = ridgeReg(np_train_dat, np_train_label, lamda)
    wt = params[0]
    pred_train = np.matmul(np.transpose(np_train_dat), wt)
    pred_train = pred_train + params[1]     # adding bias term
    rmse = calc_rmse(pred_train, np_train_label)
    rmse_train.append(rmse)
    pred_val = np.matmul(np.transpose(np_val_dat), wt)
    pred_val = pred_val + params[1]
    rmse = calc_rmse(pred_val, np_val_label)
    rmse_val.append(rmse)
    # now taking rmse for loocv
    loocv = params[3]
    loocv2 = np.array(loocv) * np.array(loocv)
    loocv_mean = np.sum(loocv2)/len(loocv)
    rmse = sqrt(loocv_mean)
    rmse_loocv.append(rmse)
    


# Modeling the training + validation data
train = np.append(np_train_dat, np_val_dat, axis = 1)
#train = np.append(train, np_test_dat.T, axis = 1)
target = np.append(np_train_label, np_val_label, axis = 0)


# removing low variance
traint = train.T
thresholder = VarianceThreshold(threshold = 0.00005)
ab = thresholder.fit_transform(traint)
train = ab[:10000, :].T
np_test_dat = ab[10000:, :]
params = ridgeReg(train, target, 0.01)




# plotting correlation plot
#dataframe = np.append(train.T, target, axis = 1)
#dataframe = pd.DataFrame(data = dataframe)
#subset = dataframe.iloc[:, 1:20]
#subset.insert(19, '20', value = dataframe.iloc[:, -1])
#corr = subset.corr()
#corr.style.background_gradient(cmap = 'coolwarm')
#plt.matshow(subset.corr())
#plt.show()









# Predicting on the test data
test_data = pd.read_csv('testData_new.csv', header = None)
test_data = test_data.iloc[:, 1:]
np_test_dat = test_data.values
wt = params[0]
bias = params[1]
prediction = np.matmul(np_test_dat, wt)
# adding bias to the values
prediction = prediction + bias

prediction[prediction < 80] = 80
prediction[prediction > 100] = 100

# creating my first submission
id_no = np.array([i for i in range(prediction.shape[0])]).T
id_no = np.reshape(id_no, (4749, 1))
submission = np.append(id_no, prediction, axis = 1).astype(int)
df = pd.DataFrame(data = submission, columns = ['Id', 'Expected'])
df.to_csv('submission.csv', index = False)

