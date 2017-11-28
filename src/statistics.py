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
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fn_train_observed_labels = defs.fn_train_observed_labels
fn_train_experiment_ids = defs.fn_train_experiment_ids
fn_train_covariates_dedim = defs.fn_train_covariates_dedim
fn_test_covariates_dedim = defs.fn_test_covariates_dedim

train_covariates = None # pandas DataFrame
train_labels = None # pandas DataFrame
train_expids = None # pandas DataFrame
test_covariates = None # pandas DataFrame

DATA_SPLIT_SHUFFLE = True
DATA_SPLIT_RANDOM_TRAINING_DATA_RATIO = 0.75


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
  experiment_indices = {}
  for i in range(len(train_expids.values)):
    if train_expids.values[i,0] not in experiments:
      experiments[train_expids.values[i,0]] = {}
    if train_labels.values[i,0] not in experiments[train_expids.values[i,0]]:
      experiments[train_expids.values[i,0]][train_labels.values[i,0]] = 0
    experiments[train_expids.values[i,0]][train_labels.values[i,0]] += 1

    if train_expids.values[i,0] not in experiment_indices:
      experiment_indices[train_expids.values[i,0]] = []
    experiment_indices[train_expids.values[i,0]].append(i)

  if verbose:
    print('experiments (%d)' % len(experiments))
    for expid in experiments:
      print('%s:' % expid)
      for label in experiments[expid]:
        print('  - %s: %d' % (label, experiments[expid][label]))

  return experiments, experiment_indices


def construct_labels(verbose=False):
  load_datasets()

  labels = {}
  label_indices = {}
  for i in range(len(train_labels.values)):
    if train_labels.values[i,0] not in labels:
      labels[train_labels.values[i,0]] = {}
    if train_expids.values[i,0] not in labels[train_labels.values[i,0]]:
      labels[train_labels.values[i,0]][train_expids.values[i,0]] = 0
    labels[train_labels.values[i,0]][train_expids.values[i,0]] += 1

    if train_labels.values[i,0] not in label_indices:
      label_indices[train_labels.values[i,0]] = []
    label_indices[train_labels.values[i,0]].append(i)

  if verbose:
    print('labels (%d)' % len(labels))
    for label in labels:
      print('%s:' % label)
      for expid in labels[label]:
        print('  - %s: %d' % (expid, labels[label][expid]))

  return labels, label_indices


def output_dataset_statistics_matrix(savepath=None):
  experiments, experiment_indices = construct_experiments()
  labels, label_indices = construct_labels()

  nExpid = []
  nLabel = []
  stat_matrix = []

  for expid in experiments:
    nExpid.append(expid)

  for label in labels:
    nLabel.append(label)

  for i in range(len(nExpid)):
    stat_matrix.append([])
    for j in range(len(nLabel)):
      stat_matrix[i].append(0)

  for label in labels:
    for expid in labels[label]:
      stat_matrix[nExpid.index(expid)][nLabel.index(label)] = labels[label][expid]

  if savepath is not None:
    with open(savepath, 'w') as wf:
      line = 'expid'
      for i in range(len(nLabel)):
        line += '\t' + nLabel[i]
      wf.write(line + '\n')

      for i in range(len(nExpid)):
        line = str(nExpid[i])
        for j in range(len(nLabel)):
          line += '\t' + str(stat_matrix[i][j])
        wf.write(line + '\n')

  return nExpid, nLabel, stat_matrix


def construct_datasets_overlap_free():
  load_datasets()

  X_train = []
  y_train = []
  X_valid = []
  y_valid = []
  X_test = []
  y2label = []

  experiments, experiment_indices = construct_experiments()
  labels, label_indices = construct_labels()

  # find expids for training and validation datasets
  expids_train_splitted = [
    47835, 51372, 52583, 54006, 55291, 57249, 59114, 59127, 59129, 59739, 
    60066, 60297, 64002, 64960, 66734, 69926, 70240, 70713, 70930, 71585, 
    72855, 74672, 75659, 75790, 76005, 76157, 76381, 77357, 77847, 78045, 
    78140, 78521, 78845, 79374, 79457, 81275, 81903, 82174, 82187, 83948, 
    86479, 87375, 89405, 90797, 90848, 90856, 94579, 95601, 96981, 97941, 
    97955, 98969
  ]

  expids_valid_splitted = []
  for expid in experiments:
    if expid not in expids_train_splitted:
      expids_valid_splitted.append(expid)

  # find data point indices for training and validation datasets
  indices_train = []
  indices_valid = []

  for i in range(len(train_expids.values)):
    if train_expids.values[i,0] in expids_train_splitted:
      indices_train.append(i)
    else:
      indices_valid.append(i)

  if DATA_SPLIT_SHUFFLE:
    random.shuffle(indices_train)
    random.shuffle(indices_valid)

  # y2label
  for label in labels:
    y2label.append(label)

  # X_train, y_train
  for index_train in indices_train:
    X_train.append(train_covariates.values[index_train].tolist())
    y_train.append(y2label.index(train_labels.values[index_train,0]))

  # X_valid, y_valid
  for index_valid in indices_valid:
    X_valid.append(train_covariates.values[index_valid].tolist())
    y_valid.append(y2label.index(train_labels.values[index_valid,0]))

  # X_test
  X_test = test_covariates.values.tolist()

  # return results
  return X_train, y_train, X_valid, y_valid, X_test, y2label


def construct_datasets_random():
  load_datasets()

  raise NotImplementedError()

  X_train = []
  y_train = []
  X_valid = []
  y_valid = []
  X_test = []
  y2label = []

  indices_train = []
  indices_valid = []

  # construct random labelled indices
  labels, label_indices = construct_labels()
  for label in label_indices:
    random.shuffle(label_indices[label])

  # construct random training and validation indices
  #TODO

  # y2label
  for label in labels:
    y2label.append(label)

  # X_train

  # TODO


def construct_datasets(method='overlap_free'):
  X_train = None
  y_train = None
  X_valid = None
  y_valid = None
  X_test = None
  y2label = None

  if (method == 'random'):
    X_train, y_train, X_valid, y_valid, X_test, y2label = construct_datasets_random()
  elif (method == 'overlap_free'):
    X_train, y_train, X_valid, y_valid, X_test, y2label = construct_datasets_overlap_free()
  else:
    raise Exception('invalid datasets construction method')

  return X_train, y_train, X_valid, y_valid, X_test, y2label


def main():
  construct_experiments(verbose=True)
  construct_labels(verbose=True)
  #construct_datasets('overlap_free')


if __name__ == '__main__':
  main()

