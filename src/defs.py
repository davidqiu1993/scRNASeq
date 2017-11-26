#!/usr/bin/env python

"""
defs.py
General definition file.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2017, David Qiu. All rights reserved."


dir_data = '../data'
fn_train_covariates = dir_data + '/train_covariates.tsv'
fn_train_observed_labels = dir_data + '/train_observed_labels.tsv'
fn_train_experiment_ids = dir_data + '/train_experiment_ids.tsv'
fn_test_covariates = dir_data + '/test_covariates.tsv'
fn_train_covariates_dedim = dir_data + '/train_covariates_dedim.pkl'
fn_test_covariates_dedim = dir_data + '/test_covariates_dedim.pkl'


if __name__ == '__main__':
  print('defs.py: general definition file.\n  - usage: import defs')


