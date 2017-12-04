#!/usr/bin/env python

"""
run_svm.py
SVM classification.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2017, David Qiu. All rights reserved."

import defs
import statistics as st
import time
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def Y2onehots(Y, NLabels):
  Y_oh = np.zeros((Y.shape[0], NLabels))
  for i in range(Y.shape[0]):
    Y_oh[i,Y[i]] = 1

  return Y_oh


def onehots2Y(Y_oh, NLabels):
  Y = np.zeros((Y_oh.shape[0], 1))

  for i in range(Y_oh.shape[0]):
    y_oh_max = Y_oh[i,0]
    y_oh_max_j = 0
    for j in range(Y_oh.shape[1]):
      if Y_oh[i,j] > y_oh_max:
        y_oh_max = Y_oh[i,j]
        y_oh_max_j = j
    Y[i,0] = y_oh_max_j

  return Y


def Y2labels(Y, y2label):
  labels = []
  for i in range(Y.shape[0]):
    labels.append([y2label[int(Y[i])]])

  return np.array(labels)


def labels2Y(labels, y2label):
  Y = np.zeros((labels.shape[0], 1))

  for i in range(labels.shape[0]):
    Y[i,0] = y2label.index(labels[i,0])

  return Y


def main():
  # construct datasets
  print('construct datasets...')
  
  X_train, Y_train, X_valid, Y_valid, X_test, y2label = st.construct_datasets()
  X_train = np.array(X_train)[1:10000,:]
  Y_train = np.array(Y_train)[1:10000]
  X_valid = np.array(X_valid)
  Y_valid = np.array(Y_valid)
  X_test = np.array(X_test)

  NFeatures = X_train.shape[1]
  NLabels = len(y2label)

  clf = svm.SVC(decision_function_shape='ovo', kernel='rbf', verbose=True)
  clf.fit(X_train, Y_train)

  # trained model summary
  print('trained model summary:')

  pred_Train = clf.predict(X_train)
  #pred_Valid = clf.predict(X_valid)
  trainAccuracy = 1.0*np.sum(pred_Train==Y_train)/pred_Train.size
  #validAccuracy = 1.0*np.sum(pred_Valid==Y_valid)/pred_Valid.size
  print('  - trainAccuracy: %f' % trainAccuracy)
  #print('  - validAccuracy: %f' % validAccuracy)


#  # output predicted test labels
#  if SHOULD_OUTPUT_TEST_LABELS:
#    print('predict and output test labels...')
#    pred_Test = model.predict_classes(X_test, verbose=0)
#    pred_Test = np.array(pred_Test).reshape((len(pred_Test),1))
#    print('  - class predictions generated: %d' % (pred_Test.shape[0]))
#    pred_labels = Y2labels(pred_Test, y2label)
#    print('  - labels converted: %d' % (pred_labels.shape[0]))
#
#    with open(fn_test_predicted_labels, 'w') as f_test_labels:
#      f_test_labels.write('"Cell Type"\n')
#      for i in range(pred_labels.shape[0]):
#        f_test_labels.write('"%s"\n' % (str(pred_labels[i,0])))


if __name__ == '__main__':
  main()

