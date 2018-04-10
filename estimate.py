import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from parse import *
from parse_test import *
from genetics import GA
from mysrc import *


tr_path = 'data/train.csv'
ts_path = 'data/test.csv'
tsy_path = 'data/test_y.csv'

sample_genes = np.load('trained_genes.npy')

regr = linear_model.LinearRegression()
Xtr_new, ytr_new = readDataTrain(tr_path)
Xtr_new, ytr_new = extract_gene(Xtr_new, ytr_new, sample_genes, True)
regr.fit(Xtr_new,ytr_new)
yhat = regr.predict(Xtr_new)
err_rate = feval(yhat, ytr_new)
print("Error rate on Training data:", err_rate)

Xts, yts = readDataTest(ts_path, tsy_path)
Xts, yts = extract_gene(Xts, yts, sample_genes, True)
yhat = regr.predict(Xts)
err_rate = feval(yhat, yts)
print("Error rate on Test data:", err_rate)
