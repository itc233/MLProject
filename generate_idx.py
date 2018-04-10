import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, metrics
from parse import read_data
from genetics import GA

def generate_idx(port_tr, port_vald, data_len):
	tr_len = np.int(port_tr*data_len)
	vald_len = np.int(port_vald*data_len) 
	rand_idx = np.arange(data_len)
	np.random.shuffle(rand_idx)
	tr_idx = rand_idx[:tr_len]
	vald_idx = rand_idx[tr_len:]
	np.save("train_idx.txt", tr_idx)
	np.save("vald_idx.txt", vald_idx)
	return tr_idx, vald_idx

def feval(yhat, y):
	fit = np.sqrt(np.mean(abs(y-yhat)**2))
	return fit


def hw_test(Xs, y, Xts, yts):
	npen = 20
	C_test = np.logspace(-1,3,npen)
	nfold = 10
	kf = KFold(n_splits=nfold,shuffle=True)
	err_rate = np.zeros((npen,nfold))
	num_nonzerocoef = np.zeros((npen,nfold))
	logreg = linear_model.LogisticRegression(penalty='l1',warm_start=True)
	for ifold, Ind in enumerate(kf.split(Xs)):
		Itr, Its = Ind
		Xtr = Xs[Itr,:]
		ytr = y[Itr]
		Xvald = Xs[Its,:]
		yvald = y[Its]
		for ipen, c in enumerate(C_test):
			logreg.C= c
			logreg.fit(Xtr, ytr)
			yhat = logreg.predict(Xvald)
			err_rate[ipen,ifold] = np.mean(yhat != yvald)
			num_nonzerocoef[ipen,ifold]=np.sum(abs(logreg.coef_[0,:])>0.001)
	err_mean = np.mean(err_rate, axis=1)
	num_nonzerocoef_mean = np.mean(num_nonzerocoef, axis=1)
	imin = np.argmin(err_mean)
	err_se = np.std(err_rate,axis=1)/np.sqrt(nfold-1)
	err_tgt = err_mean[imin] + err_se[imin]
	iopt = np.where(err_mean < err_tgt)[0][0]
	C_opt = C_test[iopt]
	logreg.C= C_opt
	logreg.fit(Xs,y)
	yhat = logreg.predict(Xts)
	err = np.mean(yhat != yts)
	return num_nonzerocoef_mean[iopt], err