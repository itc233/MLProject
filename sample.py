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
tsx_path = 'data/test.csv'
tsy_path = 'data/test_y.csv'

Xtr, ytr = readDataTrain(tr_path)
Xts, yts = readDataTest(tsx_path, tsy_path)
X = np.concatenate((Xtr, Xts), axis = 0)
X = preprocessing.scale(X)
Xtr = X[:len(ytr), :]
Xts = X[len(ytr):, :]
np.save('data/train', np.column_stack([Xtr, ytr]))
np.save('data/test', np.column_stack([Xts, yts]))

Xtr, ytr, Xvald, yvald = sample_data(Xtr, ytr, port_tr, port_vald)
GAtrain = np.column_stack([Xtr, ytr])
GAvalid = np.column_stack([Xvald, yvald])

(sample_result, sample_genes, sample_scores) = GA(GAtrain, GAvalid, linear_model.LinearRegression(), feval, iter=150, r_sample=0.5, r_crossover=0.5, r_vary=0, r_keep_best = 0.01, popsize = 300, verbose = True).select()

print("sample_result:\n", sample_result)
print("sample_genes:\n", sample_genes)
print("sample_scores:\n", sample_scores)

np.save('trained_genes', sample_genes)
