import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

def readDataTest(file_tsx, file_tsy):
	x = []
	n_row = 0
	x_row = []
	text = open(file_tsx, 'r', encoding='big5') 
	row = csv.reader(text , delimiter=",")
	for r in row:
		for i in range(2,11):
			if r[i] != "NR":
				x_row.append(float(r[i]))
			else:
				x_row.append(float(0))	
		n_row = n_row+1
		if n_row%18 == 0:
			if len(x) == 0:
				x = np.array(x_row)
			else:
				x = np.concatenate((x, np.array(x_row)), axis = 0)
			x_row = []
	text.close()

	y = []
	text = open(file_tsy, 'r', encoding='big5') 
	row = csv.reader(text , delimiter=",")
	n_row = 0
	for r in row:
		if n_row != 0:
			y.append(r[1])
		n_row = n_row+1
	text.close()

	y = np.array(y)
	print(x[0:3], np.shape(x))
	print(y[0:3], np.shape(y))
	return x, y
