import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing
# Example

#from genetics import GA

df = pd.read_excel("https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls", index_col = 0)
df1 = df.fillna(df.mean())
ystr = df1['class'].values
vals, y = np.unique(ystr, return_inverse=True)
xnames = df1.columns[:-4]
X = np.array(df1[xnames].values)
Xs = preprocessing.scale(X)
#ys = preprocessing.scale(y)
ys = y
print(ys[:10])

tr_len = np.int(0.6*len(ys))
ts_len = np.int(0.2*len(ys))
vald_len = np.int(0.2*len(ys)) 
rand_idx = np.arange(len(ys))
np.random.shuffle(rand_idx)
print(tr_len, rand_idx[0:10])
tr_idx = rand_idx[:tr_len]
Xtr = Xs[tr_idx, :]
ytr = ys[tr_idx]
Xvald = Xs[rand_idx[tr_len:tr_len+vald_len], :]
yvald = ys[rand_idx[tr_len:tr_len+vald_len]]
Xts = Xs[rand_idx[tr_len+vald_len:], :]
yts = ys[rand_idx[tr_len+vald_len:]]

print(np.shape(Xtr), np.shape(ytr))
train = np.column_stack([Xtr, ytr])
valid = np.column_stack([Xvald, yvald])

# the sample_result is a 2D numpy matrix, which is the result after sampling
# the sample_genes is the gene used for selecting instances, just ignore it if you don't need it
# the sample_scores is the final score when doing validation in valid set
def feval(y, yhat):
	err = np.sum(y != yhat)
	err_rate = err/len(y)
	return err_rate
	
#(sample_result, sample_genes, sample_scores) = GA(train, valid, linear_model.LogisticRegression(), feval).select(axis = 1)

#print("sample_result:\n", sample_result)
#print("sample_genes:\n", sample_genes)
#print("sample_scores:\n", sample_scores)

regr = linear_model.LogisticRegression()
sample_genes = [False, True, True, False, True, False, True, True, True, True, True, False, True, False, True, True, True, True, True, False, True, True, True, True, True, True, False, False, True, True, True, False, True, True, True, True, True, True, True, True, False, True, True, True, False, True, True, True ,True, False, False, True, True, True, True, True, True, True, False, True, True, True, True, False, False, True, True, True, False, False, True, False, True, True, True, True, True]
tr_idx_new = rand_idx[:tr_len+vald_len]
Xtr_new = Xs[tr_idx_new]
Xtr_new = Xtr_new[:, sample_genes]
ytr_new = ys[rand_idx[:tr_len+vald_len]]
Xts_new = Xts[:, sample_genes]
regr.fit(Xtr_new,ytr_new)
yhat = regr.predict(Xts_new)
err_rate = feval(yts, yhat)
print("Error rate on Test data:", err_rate)
yhat = regr.predict(Xtr_new)
err_rate = feval(ytr_new, yhat)
print("Error rate on Training data:", err_rate)
