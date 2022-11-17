#!/usr/bin/env python3

import numpy as np

from sklearn.datasets import load_svmlight_file
from online_k_means import online_k_means

from sklearn.cluster import KMeans as k_means
from sklearn.neighbors import KDTree as kdtree


from joblib import Parallel, delayed

def process(i):
    instance = online_k_means(k_target = 20 + i*5)

    for i in range(data.shape[0]):
        instance.process(np.asarray(data[i].todense()))
    centorid = instance.output()

    n_clusters=centorid.shape[0]
    return n_clusters



dataset = [
'a9a.txt', #adult
'ijcnn1',
'shuttle.scale',
'w8a.txt', #w8all
'letter.scale',
'poker',
'skin_nonskin.txt' #skin
]

data_path = './km_data/'
for  j in dataset:

    csr_data = load_svmlight_file(data_path + j)

    data = csr_data[0]
    print(data.shape)

    n_thread = 9
    results = Parallel(n_jobs=n_thread)(delayed(process)(i) for i in range(n_thread))

    print( j, results)





#print(data[0], data[1])