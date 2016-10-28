## (1) Import packages
import numpy as np
import pandas as pd
import sys
import csv

## (2) Read training data
train = pd.read_csv(sys.argv[1], header = None)
# Delete column 0
train = train.drop(train.columns[[0]],1)
train.columns = range(train.shape[1])

## (3) Compute statistics
# Prior
p_spam = float(sum(train.ix[:,57])) / train.shape[0]
p_nonspam = 1 - p_spam
# Estimate mean and standard deviation for every features (0, 1, ..., 56)
train_spam = train[train.ix[:,57]==1]
means_spam = train_spam.mean()
std_spam = train_spam.std()
train_nonspam = train[train.ix[:,57]==0]
means_nonspam = train_nonspam.mean()
std_nonspam = train_nonspam.std()

## (4) Data cleansing: remove features with standard deviation = 0 (in each class)
rm_idx = [list(train.columns[std_spam == 0]),list(train.columns[std_nonspam == 0])]
rm_idx = list(set([item for sublist in rm_idx for item in sublist]))

## (5) Write above statistics and rm_idx into a function ('model')
myList = [p_spam, p_nonspam, list(means_spam), list(std_spam), list(means_nonspam), list(std_nonspam), rm_idx]
myfile = open(sys.argv[2], 'wb')
for item in myList:
    myfile.write("%s\n" % item)
myfile.close()