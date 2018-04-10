import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from parse import read_data
from genetics import GA
from generate_idx import generate_idx, feval


X, y = read_data('train.csv')
X = preprocessing.scale(X)
y = preprocessing.scale(y)
tr_idx, vald_idx = generate_idx(0.7, 0.3, len(y))
Xtr = X[tr_idx, :]
ytr = y[tr_idx]
Xvald = X[vald_idx, :]
yvald = y[vald_idx]

print(np.shape(Xtr), np.shape(ytr))
train = np.column_stack([Xtr, ytr])
valid = np.column_stack([Xvald, yvald])

(sample_result, sample_genes, sample_scores) = GA(train, valid, linear_model.LinearRegression(), feval, iter=200, r_sample=0.9, r_crossover=0.5, r_vary=0.05, verbose = True).select(axis = 1)

print("sample_result:\n", sample_result)
print("sample_genes:\n", sample_genes)
print("sample_scores:\n", sample_scores)

regr = linear_model.LinearRegression()
Xtr_new = np.concatenate((Xtr, Xvald), axis = 0)
ytr_new = np.concatenate((ytr, yvald), axis = 0)
regr.fit(Xtr_new,ytr_new)

Xts, yts = read_data('test.csv')
Xts = Xts[:, sample_genes]
yhat = regr.predict(Xts)
err_rate = feval(yhat, yts)
print("Error rate on Test data:", err_rate)

