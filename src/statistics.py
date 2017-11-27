#!/usr/bin/env python

"""
statistics.py
Statistics for the datasets.
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

fn_train_observed_labels = defs.fn_train_observed_labels
fn_train_experiment_ids = defs.fn_train_experiment_ids
fn_train_covariates_dedim = defs.fn_train_covariates_dedim
fn_test_covariates_dedim = defs.fn_test_covariates_dedim

train_covariates = None
train_labels = None
train_expids = None
test_covariates = None


def load_datasets():
  global train_covariates
  global train_labels
  global train_expids
  global test_covariates

  if train_covariates is None:
    train_covariates = pd.read_pickle(fn_train_covariates_dedim)

  if train_labels is None:
    train_labels = pd.read_table(fn_train_observed_labels)

  if train_expids is None:
    train_expids = pd.read_table(fn_train_experiment_ids)

  if test_covariates is None:
    test_covariates = pd.read_pickle(fn_test_covariates_dedim)


def construct_experiments(verbose=False):
  load_datasets()

  experiments = {}
  for i in range(len(train_expids.values)):
    if train_expids.values[i,0] not in experiments:
      experiments[train_expids.values[i,0]] = {}
    if train_labels.values[i,0] not in experiments[train_expids.values[i,0]]:
      experiments[train_expids.values[i,0]][train_labels.values[i,0]] = 0
    experiments[train_expids.values[i,0]][train_labels.values[i,0]] += 1

  if verbose:
    print('experiments (%d)' % len(experiments))
    for expid in experiments:
      print('%s:' % expid)
      for label in experiments[expid]:
        print('  - %s' % label)


def construct_labels(verbose=False):
  load_datasets()

  labels = {}
  for i in range(len(train_labels.values)):
    if train_labels.values[i,0] not in labels:
      labels[train_labels.values[i,0]] = {}
    if train_expids.values[i,0] not in labels[train_labels.values[i,0]]:
      labels[train_labels.values[i,0]][train_expids.values[i,0]] = 0
    labels[train_labels.values[i,0]][train_expids.values[i,0]] += 1

  if verbose:
    print('labels (%d)' % len(labels))
    for label in labels:
      print('%s:' % label)
      for expid in labels[label]:
        print('  - %s' % expid)


def main():
  construct_experiments(verbose=True)
  construct_labels(verbose=True)


if __name__ == '__main__':
  main()

