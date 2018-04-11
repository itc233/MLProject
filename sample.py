import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from parse import *
from parse_test import *
from genetics import GA
from mysrc import *


port_tr = 0.7
port_vald = 0.3
tr_path = 'data/train.csv'
ts_path = 'data/test.csv'

Xtr, ytr = readDataTrain(tr_path)
Xts, yts = readDataTest(ts_path)
X = np.concatenate((Xtr, Xts), axis = 0)
X = preprocessing.scale(X)
Xtr = X[:len(ytr), :]
Xts = X[len(ytr):, :]
np.save('data/train', np.concatenate((Xtr, ytr)), axis = 1)
np.save('data/test', np.concatenate((Xts, yts)), axis = 1)

GAtrain = np.column_stack([Xtr, ytr])
GAvalid = np.column_stack([Xvald, yvald])

(sample_result, sample_genes, sample_scores) = GA(GAtrain, GAvalid, linear_model.LinearRegression(), feval, iter=300, r_sample=0.9, r_crossover=0.5, r_vary=0.05, verbose = True).select(axis = 1)

print("sample_result:\n", sample_result)
print("sample_genes:\n", sample_genes)
print("sample_scores:\n", sample_scores)

np.save('trained_genes', sample_genes)
