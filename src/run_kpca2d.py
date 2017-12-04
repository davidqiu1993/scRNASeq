#!/usr/bin/env python

"""
run_kpca2d.py
Kernel pincipal components analysis (kPCA) example transfering a subset of 
training input data into two-dimensional principal components.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2017, David Qiu. All rights reserved."

import defs
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mlxtend.* import *
from sklearn.decomposition import PCA, KernelPCA

fn_train_covariates = defs.fn_train_covariates
fn_train_observed_labels = defs.fn_train_observed_labels
fn_train_experiment_ids = defs.fn_train_experiment_ids
fn_test_covariates = defs.fn_test_covariates


def main():
  chunksize = 1000
  X = None
  y = None

  t0 = time.time()

  with open(fn_train_covariates, 'r') as tsv:
    df_reader = pd.read_table(tsv, chunksize=chunksize)
    for df in df_reader:
      kpca = KernelPCA(n_components=2, kernel="rbf", fit_inverse_transform=True, gamma=10)
      kpca.fit(df)
      X = kpca.transform(df)
      break

  with open(fn_train_observed_labels, 'r') as tsv:
    df_reader = pd.read_table(tsv, chunksize=chunksize)
    for df in df_reader:
      y = df.values
      break

  t1 = time.time()

  print('runtime = %f sec' % (t1 - t0))

  #pd.DataFrame(X).to_pickle('test_train_covariates_dedim.pkl')

  #print(pd.read_pickle('test_train_covariates_dedim.pkl'))


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

