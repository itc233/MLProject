import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from parse import *
from parse_test import *
from genetics import GA
from mysrc import *
import csv

tr_path = 'trainDataForGA.npy'
ts_path = 'testDataForGA.npy'
sample_genes = np.load('trained_genes.npy')

regr = linear_model.Lasso(alpha = 0.01)#LinearRegression()

Xytr = np.load(tr_path)#readDataTrain(tr_path)
Xyts = np.load(ts_path)#readDataTrain(tr_path)
Xyts = Xyts.astype(float)
Xtr = Xytr[:, :-1]
ytr = Xytr[:, -1]
Xtr = Xtr[:, sample_genes]
Xts = Xyts[:, :-1]
yts = Xyts[:, -1]
Xts = Xts[:, sample_genes]

regr.fit(Xtr,ytr)
yhat = regr.predict(Xtr)
err_rate = np.sqrt(np.mean((ytr-yhat)**2))
print("Error rate on Training data:", err_rate)


yhat = regr.predict(Xts)
err_rate = np.sqrt(np.mean((yts-yhat)**2))
print("Error rate on Test data:", err_rate)

#parse output
output_path = 'data/forkaggle.csv'
with open(output_path, 'w') as f:
        f.write('id,value\n')
        for i, v in  enumerate(yhat):
            f.write('id_%d,%d\n' %(i, v))
