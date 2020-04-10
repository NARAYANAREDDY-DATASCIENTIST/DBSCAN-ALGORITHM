# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:17:21 2020

@author: NARAYANA REDDY DATA SCIENTIST
"""
# DBSCAN CLUSTERING
# IMPORT THE LIBRARIES
import numpy as np
import pandas as pd

# IMPORT THE DATASET
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

# BUILD THE DBSCAN ALGORITHM
from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=3,min_samples=4)

# FITTING THE MODEL
model=dbscan.fit(x)
labels=model.labels_

# IDENTIFYING  THE POINTS WHICH MAKES UP OUR CORE POINTS

sample_cores=np.zeros_like(labels,dtype=bool)
sample_cores[dbscan.core_sample_indices_]=True


# CALCULATING THE NUMBER OF CLUSTERS
n_clusters=len(set(labels))-(1 if -1 in labels else 0)


