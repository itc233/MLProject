import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn import metrics
from parse import *
from parse_test import *
from genetics import GA
from mysrc import *


tr_path = 'data/train.npy'
ts_path = 'data/test.npy'
#tsy_path = 'data/test_y.csv'

sample_genes = np.load('trained_genes.npy')

regr = linear_model.LinearRegression()
Xytr = np.load(tr_path)#readDataTrain(tr_path)
Xtr = Xytr[:, :-1]
ytr = Xytr[:, -1]
Xtr = Xtr[:, sample_genes]
regr.fit(Xtr,ytr)
yhat = regr.predict(Xtr)
err_rate = feval(yhat, ytr)
print("Error rate on Training data:", err_rate)

Xyts = np.load(ts_path)#readDataTrain(tr_path)
Xyts = Xyts.astype(float)
Xts = Xyts[:, :-1]
yts = Xyts[:, -1]
Xts = Xts[:, sample_genes]
yhat = regr.predict(Xts)
err_rate = feval(yhat, yts)
print("Error rate on Test data:", err_rate)
