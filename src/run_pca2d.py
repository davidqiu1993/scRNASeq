#!/usr/bin/env python

"""
main.py
Main script for single-cell RNA sequence classfication application.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2017, David Qiu. All rights reserved."

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mlxtend.* import *
from sklearn.decomposition import PCA

dir_data = '../data'
fn_train_covariates = dir_data + '/train_covariates.tsv'
fn_train_observed_labels = dir_data + '/train_observed_labels.tsv'
fn_train_experiment_ids = dir_data + '/train_experiment_ids.tsv'
fn_test_covariates = dir_data + '/test_covariates.tsv'


def main():
  chunksize = 1000
  X = None
  y = None

  t0 = time.time()

  with open(fn_train_covariates, 'r') as tsv:
    df_reader = pd.read_table(tsv, chunksize=chunksize)
    for df in df_reader:
      pca = PCA(n_components=2)
      pca.fit(df)
      X = pca.transform(df)
      break

  with open(fn_train_observed_labels, 'r') as tsv:
    df_reader = pd.read_table(tsv, chunksize=chunksize)
    for df in df_reader:
      y = df.values
      break

  t1 = time.time()

  print('runtime = %f sec' % (t1 - t0))


  #print(X)
  #print(y)

  y_tags = []
  for yelem in y:
    if yelem[0] not in y_tags:
      y_tags.append(yelem[0])

  fig, ax = plt.subplots()
  ax.scatter(X[:,0], X[:,1])

  #print(y_tags)
  for i, yelem in enumerate(y):
    #print(yelem)
    ax.annotate(str(y_tags.index(yelem[0])), (X[i,0], X[i,1]))

  plt.show()


if __name__ == '__main__':
  main()

