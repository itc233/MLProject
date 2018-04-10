import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

def readDataTest(file_tsx, file_tsy):
	x = []
	for i in range(18):
		x.append([])
	n_row = 0
	text = open(file_tsx, 'r', encoding='big5') 
	row = csv.reader(text , delimiter=",")
	for r in row:
		if n_row != 0:
			for i in range(2,10):
				if r[i] != "NR":
					x[(n_row-1)%18].append(float(r[i]))
				else:
					x[(n_row-1)%18].append(float(0))	
		n_row = n_row+1
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
	# 每 12 個月
	x = np.array(x)
	y = np.array(y)
	print(x[0:10])
	print(y[0:10])
	return x, y
