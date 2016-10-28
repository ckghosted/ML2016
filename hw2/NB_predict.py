## (1) Import packages
import numpy as np
import pandas as pd
import sys
import csv
import re
import string

## (2) Read model
with open(sys.argv[1], 'rb') as f:
    reader = csv.reader(f)
    myList = list(reader)
p_spam = float(''.join(ch for ch in myList[0] if ch not in '\[|\]'))
p_nonspam = float(''.join(ch for ch in myList[1] if ch not in '\[|\]'))
means_spam = [float(''.join(ch for ch in string if ch not in '\[|\]')) for string in myList[2]]
std_spam = [float(''.join(ch for ch in string if ch not in '\[|\]')) for string in myList[3]]
means_nonspam = [float(''.join(ch for ch in string if ch not in '\[|\]')) for string in myList[4]]
std_nonspam = [float(''.join(ch for ch in string if ch not in '\[|\]')) for string in myList[5]]
rm_idx = [int(''.join(ch for ch in string if ch not in '\[|\]')) for string in myList[6]]

## (3) Read testing data
test = pd.read_csv(sys.argv[2], header = None)
# Delete column 0
test = test.drop(test.columns[[0]],1)
test.columns = range(test.shape[1])

## (4) Define functions for Gaussian PDF and all kinds of conditional probabilities
def gaussianPDF(x, mean, std):
    return (1 / (np.sqrt(2*np.pi) * std)) * np.exp(-(x-mean)**2/(2*std**2))
def probGiven0(x, dim):
    return gaussianPDF(x, means_nonspam[dim], std_nonspam[dim]) * p_nonspam
def probGiven1(x, dim):
    return gaussianPDF(x, means_spam[dim], std_spam[dim]) * p_spam
def prob0Givenx(x, dim):
    return probGiven0(x, dim) / (probGiven0(x, dim) + probGiven1(x, dim))
def prob1Givenx(x, dim):
    return probGiven1(x, dim) / (probGiven0(x, dim) + probGiven1(x, dim))

## (5) Prediction
result = list()
for i in range(test.shape[0]):
    prob0 = 1
    prob1 = 1
    for j in range(test.shape[1]):
        if j in rm_idx:
            continue
        prob0 *= prob0Givenx(test.ix[i,j], j)
        prob1 *= prob1Givenx(test.ix[i,j], j)
    result.append(1 if prob0 < prob1 else 0)
result = pd.Series(result)
result.index = range(1,601)
result.to_csv(sys.argv[3], index_label = ['id'], header = ['label'])
