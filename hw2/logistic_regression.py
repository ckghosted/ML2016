## (1) Import packages
import numpy as np
import pandas as pd
import sys

## (2) Read training data
train = pd.read_csv(sys.argv[1], header = None)
# Delete column 0
train = train.drop(train.columns[[0]],1)

## (3) Create training feature DataFrame and outcome Series
# Split features and outcome
y = pd.Series(train.ix[:, train.shape[1]])
del train[58]
# Simple 1st-order linear regression
## Normalize
train_norm = (train - train.mean()) / train.std()
# train_norm = train
## add constant 1
train_norm.insert(loc = 0, column = 0, value = 1)

## (4) Define the sigmoid function
def sigmoid(z):
    return [1.0/(1.0+np.exp(-zi)) for zi in z]

## (5) Define the loss function for logistic regression
def computeLoss(X, y, coef):
    return -sum(np.log(sigmoid(y.multiply(np.dot(X, coef)))))

## (6) Define the loss function for logistic regression with regularization
def computeLoss_reg(X, y, coef, lb):
    return -sum(np.log(sigmoid(y.multiply(np.dot(X, coef))))) + lb * 0.5 * sum(coef[1:])

## (7) Gradient descent
#Set parameters
iteration = 200
eta = 0.003
alpha = eta # fixed learning rate
loss_history = list()
# Initialize coefficients to be all 0
coef = pd.Series(np.repeat(0.0, train_norm.shape[1], axis=0))
# Main loop of gradient descent
for ite in range(iteration):
    loss_history.append(computeLoss(train_norm, y, coef))
    temp = list()
    for i in range(train_norm.shape[1]):
        temp.append(coef[i] - alpha * sum((sigmoid(np.dot(train_norm, coef)) - y).multiply(train_norm.ix[:,i])))
    for i in range(train_norm.shape[1]):
        coef[i] = temp[i]
coef_save = pd.DataFrame(coef)
coef_save = pd.concat([coef_save.reset_index(drop = True), train.mean(), train.std()], axis=1)
coef_save.to_csv(sys.argv[2], index_label = ['id'], header = ['coef', 'train_mean', 'train_std'])
