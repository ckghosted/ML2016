## (1) Import packages
import numpy as np
import pandas as pd
import sys

## (2) Read model (e.g., coefficients of logistic regression)
model = pd.read_csv(sys.argv[1], header = 0)
model = model.drop(model.columns[[0]],1)
# model
coef = model.ix[:,0]
train_mean = model.ix[1:,1]
train_std = model.ix[1:,2]

## (3) Read testing data
test = pd.read_csv(sys.argv[2], header = None)
# Delete column 0
test = test.drop(test.columns[[0]],1)

## (4) Testing data normalization
test_norm = (test - train_mean) / train_std
# Add all 1
test_norm.insert(loc = 0, column = 0, value = 1)

## (5) Define the sigmoid function
def sigmoid(z):
    return [1.0/(1.0+np.exp(-zi)) for zi in z]

## (6) Predict
result = pd.Series([1 if x > 0.5 else 0 for x in sigmoid(test_norm.dot(coef))])
result.index = range(1,601)
result.to_csv(sys.argv[3], index_label = ['id'], header = ['label'])