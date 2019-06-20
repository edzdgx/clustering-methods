#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# file to read
infile = 'turbine_data.csv'
df = pd.read_csv(infile, sep=',',header=None)
speedpower = df.values

# perform DBSCAN clustering
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(speedpower)
model = DBSCAN(eps=0.1, min_samples=20).fit_predict(x)

plt.scatter(x[:, 0], x[:, 1], c=model)
plt.title("DBSCAN Cluster")
plt.savefig('DBSCAN.png')
plt.show()