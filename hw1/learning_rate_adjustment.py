## (1) Import packages
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import sys
%matplotlib inline

## (2) Read training data
train = pd.read_csv('train.csv', encoding = 'big5')
# Change column names
train.columns = ['date','station','item','0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
# Delete column 'station'
train = train.drop(train.columns[[1]],1)

## (3) Reshape the training data to be one value per row
train_melt = pd.melt(train, id_vars=['date', 'item'], value_vars=[str(s) for s in list(range(24))])
train_melt['date'] = pd.to_datetime(train_melt['date'])
train_melt['variable'] = pd.to_numeric(train_melt['variable'])
# Convert all values into float and set 'NR' in 'RAINFALL' to be 0.0
train_melt['value'] = pd.to_numeric(train_melt['value'], errors = 'coerce')
train_melt.fillna(0, inplace = True)
# Sort all values according to 'date' and 'variable' (hour in 0~23)
train_melt.sort_values(by = ['date', 'variable'], ascending = [1, 1], inplace = True)
train_melt.index = range(train_melt.shape[0])

## (4) Extract all 10-hour window
# Since there are 18 items, start from 0 and then 18, ... and so on
for i in range(0, train_melt.shape[0]-1, 18):
    if i + 171 > train_melt.shape[0]-1:
        break
    elif (train_melt.iloc[i+171,0].replace(hour = train_melt.iloc[i+171,2]) - train_melt.iloc[i,0].replace(hour = train_melt.iloc[i,2])) / np.timedelta64(1, 'h') > 9:
        continue
    tmp = train_melt.iloc[range(i,i+162) + [i+171], 3]
    if i == 0:
        train_all = tmp
    else:
        train_all = pd.concat([train_all.reset_index(drop=True), tmp.reset_index(drop=True)], axis=1)
train_all = train_all.T
train_all.index = range(train_all.shape[0])

## (5) Set feature names (and let outcome to be 'y')
featureNames = list()
for hr in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9']:
    for var in train.ix[0:17, 1]:
        featureNames.append(var + '_' + hr)
featureNames.append('y')
train_all.columns = featureNames

## (6) Create training feature DataFrame and outcome Series
# Remove wrong data or outlier
# Just remove all records with any feature < 0
train_clean = train_all[(train_all >= 0).all(1)]
# Shuffle before split features and outcome
train_clean = train_clean.reindex(np.random.permutation(train_clean.index))
train_clean.index = range(train_clean.shape[0])
# Split features and outcome
y = pd.Series(train_clean.ix[:, train_clean.shape[1]-1])
del train_clean['y']
# Simple 1st-order linear regression
## Normalize
train_norm = (train_clean - train_clean.mean()) / train_clean.std()
## add constant 1
train_norm.insert(loc = 0, column = 'intercept', value = 1)

## (7) Define the loss function
def computeLoss(X, y, coef):
    y_hat = np.dot(X, coef)
    return sum((y - y_hat)**2) / len(y)

## (8) Visualize Learning Rate (GD)
eta_list = [0.000006, 0.000008, 0.00001, 0.000012]
iteration = 20
loss_for_LR = list()
for alpha in eta_list:
    random.seed(1002)
    coef_temp = [random.random()/100.0 for i in range(train_norm.shape[1])]
    loss_history_temp = list()
    for ite in range(iteration):
        loss_history_temp.append(computeLoss(train_norm, y, coef_temp))
        temp = list()
        for i in range(train_norm.shape[1]):
            temp.append(coef_temp[i] - alpha * sum((np.dot(train_norm, coef_temp) - y).multiply(train_norm.ix[:,i])))
        for i in range(train_norm.shape[1]):
            coef_temp[i] = temp[i]
    loss_for_LR.append(loss_history_temp)
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(eta_list)):
    ax.plot(range(iteration), loss_for_LR[i], '--', linewidth=2, label='$\eta = %s$'%eta_list[i])
ax.legend(loc = 'right', fontsize=10,fancybox=True)
ax.set_xlabel('Number of iteration')
ax.set_ylabel('Loss')