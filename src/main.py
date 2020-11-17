#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Kmeans import Kmeans

# file to read
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--infile', required=True, help='input data file')
# ap.add_argument('-o', '--outfile', required=True, help='output data file')
ap.add_argument('-c', '--cluster', default=2, type=int, help='number of clusters')
args = vars(ap.parse_args())
infile = args['infile']
cluster = args['cluster']
# outfile = args['outfile']

# read csv file, get speedpower
df = pd.read_csv(infile, sep=',',header=None)
speedpower = df.values

# instantiate Kmeans obj
X = Kmeans(df, cluster)
# Y = X.k_means()

X.init()
X.next_iter()
