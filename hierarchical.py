import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# file to read
infile = 'turbine_data.csv'
df = pd.read_csv(infile, sep=',',header=None)

# get [speed, power] into a dataframe
speed_power = df.values
X = speed_power

# perform agglomerative clustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='mahalanobis', linkage='complete')
cluster.fit_predict(X)

# save output figure
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='prism')
plt.title("Hierarchical Cluster")
plt.savefig('HierarchicalClustering.png')
plt.show()