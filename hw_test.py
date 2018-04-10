import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing, metrics
from parse import read_data
from genetics import GA

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