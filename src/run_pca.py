#!/usr/bin/env python

"""
run_pca.py
Pincipal components analysis (PCA) for training and test data sets, which 
transfers the training and test input data into 100-dimensional principal 
components.
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
from sklearn.decomposition import PCA

fn_train_covariates = defs.fn_train_covariates
fn_train_observed_labels = defs.fn_train_observed_labels
fn_train_experiment_ids = defs.fn_train_experiment_ids
fn_test_covariates = defs.fn_test_covariates
fn_train_covariates_dedim = defs.fn_train_covariates_dedim
fn_test_covariates_dedim = defs.fn_test_covariates_dedim


def main():
  X_train = None
  X_test = None
  X_join = None
  X_train_dedim = None
  X_test_dedim = None

  time_begin = None
  time_end = None


  # load train_covariates
  print('load train_covariates...')
  time_begin = time.time()

  f_train_covariates = open(fn_train_covariates, 'r')
  X_train = pd.read_table(f_train_covariates)
  f_train_covariates.close()

  time_end = time.time()
  print('  - runtime: %f sec' % (time_end - time_begin))


  # load test_covariates
  print('load test_covariates...')
  time_begin = time.time()

  f_test_covariates = open(fn_test_covariates, 'r')
  X_test = pd.read_table(f_test_covariates)
  f_test_covariates.close()

  time_end = time.time()
  print('  - runtime: %f sec' % (time_end - time_begin))


  # merge covariates
  print('merge covariates...')
  time_begin = time.time()

  X_join = pd.concat([X_train, X_test])

  time_end = time.time()
  print('  - runtime: %f sec' % (time_end - time_begin))


  # PCA on both covariates
  print('PCA on both covariates...')
  time_begin = time.time()

  n_components = 100
  pca = PCA(n_components=n_components)
  print('  - n_components: %d' % n_components)
  pca.fit(X_join)
  X_train_dedim = pca.transform(X_train)
  X_test_dedim = pca.transform(X_test)

  time_end = time.time()
  print('  - runtime: %f sec' % (time_end - time_begin))


  # save dedim files
  print('save dedim files...')
  time_begin = time.time()

  pd.DataFrame(X_train_dedim).to_pickle(fn_train_covariates_dedim)
  pd.DataFrame(X_test_dedim).to_pickle(fn_test_covariates_dedim)

  time_end = time.time()
  print('  - runtime: %f sec' % (time_end - time_begin))


if __name__ == '__main__':
  main()

