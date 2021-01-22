import numpy as np
import csv
import sys
import os
import json
import keras
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import matplotlib as mptlib
#mptlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

import settings
import preprocess



# setup kfolds
def GetSetupKfolds(floc, numfolds, idfold):
  # get id from setupfiles
  dataidsfull = []
  with open(floc, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       dataidsfull.append( int( row['dataid']))
  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds)
     allkfolds   = [ (train_index, test_index) for train_index, test_index in kf.split(dataidsfull )]
     train_all_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
     len_train = len(train_all_index)
     train_index = train_all_index[:int(0.8*len_train)]
     valid_index = train_all_index[int(0.8*len_train):]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None
     valid_index = None
  print("kfold: \t",numfolds)
  print("idfold: \t", idfold)
  print("train_index:\t", train_index)
  print("valid_index:\t", valid_index)
  print("test_index:\t",  test_index)
  return (train_index,test_index,valid_index)







###
### training option helper functions
###

def GetCallbacks(logfileoutputdir, stage):
  logdir   = logfileoutputdir+"/"+stage
  filename = logfileoutputdir+"/"+stage+"/modelunet.h5"
  logname  = logfileoutputdir+"/"+stage+"/log.csv"
  callbacks = [ keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.CSVLogger(logname),
                    keras.callbacks.ModelCheckpoint(filepath=filename, verbose=1, save_best_only=True),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=5, min_lr=0.0001),
                    keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True, write_images=False),
  ]
  return callbacks, filename

def GetOptimizer(tuning=False):
  if tuning:
      if settings.options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=0.1*settings.options.lr)
      elif settings.options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(0.1)
      elif settings.options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.0002)
      elif settings.options.trainingsolver=="sgd":
          opt = keras.optimizers.SGD(0.001)
      else:
          opt = settings.options.trainingsolver
  else:
      if settings.options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=settings.options.lr)
      elif settings.options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(1.0)
      elif settings.options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.002)
      elif settings.options.trainingsolver=="sgd":
          opt = keras.optimizers.SGD(0.01)
      else:
          opt = settings.options.trainingsolver
  return opt

def GetLoss(stage):

  from mymetrics import dsc_l2, dsc_matlab_l2, dsc_l2_liver, dsc_l2_tumor, dsc_l2_ensemble, dsc_l2_bxe

  if stage=='liver':
      lss = dsc_l2
      met = [dsc_l2]
  elif stage=='tumor':
      lss = dsc_l2
      met = [dsc_l2]
  elif stage=='ensemble':
      lss = dsc_l2_ensemble
      met = [dsc_l2_ensemble, dsc_l2_liver, dsc_l2_tumor]
  return lss, met
