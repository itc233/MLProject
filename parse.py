import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

def read_data(file):
	data = []
	for i in range(18):
		data.append([])
	n_row = 0
	text = open(file, 'r', encoding='big5') 
	row = csv.reader(text , delimiter=",")
	for r in row:
		if n_row != 0:
			for i in range(3,27):
				if r[i] != "NR":
					data[(n_row-1)%18].append(float(r[i]))
				else:
					data[(n_row-1)%18].append(float(0))	
		n_row = n_row+1
	text.close()

	x = []
	y = []
	# 每 12 個月
	for i in range(12):
		for j in range(471):
			x.append([])
			for t in range(18):
				for s in range(9):
					x[471*i+j].append(data[t][480*i+j+s])
			y.append(data[9][480*i+j+9])
	x = np.array(x)
	y = np.array(y)
	return x, y
