#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import DBSCAN

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--infile', required=True, help='input data file')
ap.add_argument('-o', '--outfile', required=True, help='output data file')
args = vars(ap.parse_args())
infile = args['infile']
outfile = args['outfile']

# read csv file, get speedpower
df = pd.read_csv(infile, sep=',',header=None)
speedpower = df.values

# perform DBSCAN clustering
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(speedpower)
model = DBSCAN(eps=0.1, min_samples=20).fit_predict(x)

# output result
plt.scatter(x[:, 0], x[:, 1], c=model)
plt.title("DBSCAN Cluster")
plt.savefig(outfile)
plt.show()