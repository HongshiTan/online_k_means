#!/usr/bin/env python3

import numpy as np

from sklearn.datasets import load_svmlight_file
from online_k_means import online_k_means;

from sklearn.cluster import KMeans as k_means;
from sklearn.neighbors import KDTree as kdtree

csr_data = load_svmlight_file("./km_data/a9a.txt")
data = csr_data[0]
print(data.shape)

instance = online_k_means(k_target = 20)

for i in range(data.shape[0]):
    instance.process(np.asarray(data[i].todense()))


centorid = instance.output()

n_clusters=centorid.shape[0]

kmeans = k_means(
    init="random",
    n_clusters=n_clusters,
    n_init=20,
    max_iter=300,
    random_state=42
)

ground_truth =  kmeans.fit(data.todense())


kd = kdtree(centorid, leaf_size=1)

d, ind = kd.query(ground_truth.cluster_centers_, k=1)

print(d)
print(ground_truth)
print(centorid)



#print(data[0], data[1])