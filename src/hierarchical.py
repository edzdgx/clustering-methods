import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--infile', required=True, help='input data file')
ap.add_argument('-o', '--outfile', required=True, help='output data file')
args = vars(ap.parse_args())
infile = args['infile']
outfile = args['outfile']

# read csv file, get speedpower
df = pd.read_csv(infile, sep=',',header=None)
print(df)
speedpower = df.values

# apply clustering algorithm to speedpower
cluster = AgglomerativeClustering(n_clusters=2, affinity='mahalanobis', linkage='complete')
cluster.fit_predict(speedpower)

# output result
plt.scatter(speedpower[:,0],speedpower[:,1], c=cluster.labels_)
plt.title("Hierarchical Cluster")
plt.savefig(outfile)
plt.show()