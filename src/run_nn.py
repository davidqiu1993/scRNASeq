#!/usr/bin/env python

"""
run_nn.py
Neural network classification.
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
#from mlxtend.* import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from keras.callbacks import EarlyStopping

fn_nn_model_weights = defs.fn_nn_model_weights

SHOULD_TRAIN_MODEL = False


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
    labels.append([y2label[Y[i]]])

  return np.array(labels)


def labels2Y(labels, y2label):
  Y = np.zeros((labels.shape[0], 1))

  for i in range(labels.shape[0]):
    Y[i,0] = y2label.index(labels[i,0])

  return Y


def buildModel(NFeatures, NLabels, summary=True):
  model = Sequential()

  model.add(Dense(input_dim=NFeatures,
                  output_dim=100,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dense(output_dim=64,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dense(output_dim=64,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dense(output_dim=64,
                  activation='relu',
                  kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dense(output_dim=NLabels,
                  activation='softmax',
                  kernel_regularizer=regularizers.l2(0.00)))
  
  model.compile(optimizer='sgd', #'adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.summary()

  return model


def main():
  # construct datasets
  print('construct datasets...')
  
  X_train, Y_train, X_valid, Y_valid, X_test, y2label = st.construct_datasets()
  X_train = np.array(X_train)
  Y_train = np.array(Y_train).reshape(len(Y_train),1)
  X_valid = np.array(X_valid)
  Y_valid = np.array(Y_valid).reshape(len(Y_valid),1)
  X_test = np.array(X_test)

  NFeatures = X_train.shape[1]
  NLabels = len(y2label)

  Y_train = Y2onehots(Y_train, NLabels)
  Y_valid = Y2onehots(Y_valid, NLabels)

  print('  - X_train: (%d, %d)' % (X_train.shape[0], X_train.shape[1]))
  print('  - Y_train: (%d, %d)' % (Y_train.shape[0], Y_train.shape[1]))
  print('  - X_valid: (%d, %d)' % (X_valid.shape[0], X_valid.shape[1]))
  print('  - Y_valid: (%d, %d)' % (Y_valid.shape[0], Y_valid.shape[1]))

  # build model
  print('bulid model...')
  model = buildModel(NFeatures, NLabels)

  # train model and save weights
  if (not os.path.isfile(fn_nn_model_weights)) or SHOULD_TRAIN_MODEL:
    print('train model...')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    history = model.fit(X_train, Y_train,
                        batch_size=256,
                        epochs=100,
                        verbose=1,
                        shuffle=True,
                        callbacks=[early_stopping],
                        validation_data=(X_valid, Y_valid))

    his_acc = history.history['acc']
    his_loss = history.history['loss']
    his_val_acc = history.history['val_acc']
    his_val_loss = history.history['val_loss']
    eps = range(len(his_acc))

    model.save_weights(fn_nn_model_weights)

    plt.figure(1)
    plt.plot(eps, his_acc, 'b-', eps, his_val_acc, 'r-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('training/validation accuracy')
    plt.grid(True)
    
    plt.figure(2)
    plt.plot(eps, his_loss, 'b-', eps, his_val_loss, 'r-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training/validation loss')
    plt.grid(True)

    plt.show()

  # load model weights
  if os.path.isfile(fn_nn_model_weights):
    print('load model weights...')
    model.load_weights(fn_nn_model_weights)

  # trained model summary
  print('trained model summary:')

  pred_Train = model.predict_classes(X_train, verbose=0)
  pred_Valid = model.predict_classes(X_valid, verbose=0)
  trainAccuracy = 1.0*np.sum(pred_Train==np.argmax(Y_train,axis=1))/pred_Train.size
  validAccuracy = 1.0*np.sum(pred_Valid==np.argmax(Y_valid,axis=1))/pred_Valid.size
  print('  - trainAccuracy: %f' % trainAccuracy)
  print('  - validAccuracy: %f' % validAccuracy)

  maxClassValidPercentage = np.max(np.sum(Y_valid,axis=0))/np.sum(Y_valid)
  maxClassTrainPercentage = np.max(np.sum(Y_train,axis=0))/np.sum(Y_train)
  print('  - maxClassTrainPercentage: %f' % maxClassTrainPercentage)
  print('  - maxClassValidPercentage: %f' % maxClassValidPercentage)


if __name__ == '__main__':
  main()

