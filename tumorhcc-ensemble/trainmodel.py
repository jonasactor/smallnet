import numpy as np
import csv
import sys
import os
import json
import keras
#from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D
#from keras.models import Model, Sequential
#from keras.models import model_from_json, load_model
#from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
#from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import matplotlib as mptlib
#mptlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

import settings
import preprocess
from generator import NpyDataGenerator_Liver, NpyDataGenerator_Tumor, NpyDataGenerator_Ensemble

###
### Training: build NN model from anonymized data
###
def TrainModel(idfold=0, saveloclist=None):

    from setupmodel import GetSetupKfolds, GetCallbacks, GetOptimizer, GetLoss
    from buildmodel import get_unet_liver, get_unet_tumor, get_unet_ensemble

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    ###
    ### set up output, logging, callbacks, and k-folds
    ###
    kfolds = settings.options.kfolds

    logfileoutputdir= '%s/%03d/%03d' % (settings.options.outdir, kfolds, idfold)
    os.system ('mkdir -p ' + logfileoutputdir)
#    os.system ('mkdir -p ' + logfileoutputdir + '/nii')
    os.system ('mkdir -p ' + logfileoutputdir + '/liver')
    os.system ('mkdir -p ' + logfileoutputdir + '/tumor')
    os.system ('mkdir -p ' + logfileoutputdir + '/tuning')
    os.system ('mkdir -p ' + logfileoutputdir + '/ensemble')
    print("Output to\t", logfileoutputdir)

    (train_index,test_index,valid_index) = GetSetupKfolds(settings.options.dbfile, kfolds, idfold)

    if settings.options.verbose:
        vb=1
    else:
        vb=2

    ########
    ########
    ###
    ### TRAIN LIVER MODEL
    ###
    ########
    ########
    saveloclist = settings.options.datafiles_liver
    loclist = np.genfromtxt(saveloclist, delimiter=',', dtype='str')[1:]
    trainingsubset = [ row for row in loclist if int(row[0]) in train_index]
    testingsubset  = [ row for row in loclist if int(row[0]) in test_index ]
    validsubset    = [ row for row in loclist if int(row[0]) in valid_index]

    opt                 = GetOptimizer()
    callbacks, modelloc = GetCallbacks(logfileoutputdir, "liver")
    lss, met            = GetLoss('liver')
    liver_model         = get_unet_liver()
    liver_model.summary()
    liver_model.compile(loss=lss, metrics=met, optimizer=opt)
    print("\n\n#\t liver model training...\tModel parameters: {0:,}".format(liver_model.count_params()))

    train_xlist = [row[1] for row in trainingsubset]
    train_ylist = [row[2] for row in trainingsubset]

    valid_xlist = [row[1] for row in validsubset]
    valid_ylist = [row[2] for row in validsubset]

    training_generator   = NpyDataGenerator_Liver(train_xlist, train_ylist)
    validation_generator = NpyDataGenerator_Liver(valid_xlist, valid_ylist)
    history_liver = liver_model.fit_generator( \
                        verbose             = vb,
                        generator           = training_generator,
                        validation_data     = validation_generator,
                        use_multiprocessing = True,
                        workers             = 16,
                        epochs              = settings.options.numepochs,
                        callbacks           = callbacks,
                        shuffle             = True,
                        )

    
    ########
    ########
    ###
    ### TRAIN TUMOR MODEL WITH TRUE MASK DATA
    ###
    ########
    ########
    saveloclist = settings.options.datafiles_tumor
    loclist = np.genfromtxt(saveloclist, delimiter=',', dtype='str')[1:]
    trainingsubset = [ row for row in loclist if int(row[0]) in train_index]
    testingsubset  = [ row for row in loclist if int(row[0]) in test_index ]
    validsubset    = [ row for row in loclist if int(row[0]) in valid_index]

    opt                 = GetOptimizer()
    callbacks, modelloc = GetCallbacks(logfileoutputdir, "tumor")
    lss, met            = GetLoss('tumor')
    tumor_model         = get_unet_tumor(liver_model)
    tumor_model.summary()
    tumor_model.compile(loss=lss, metrics=met, optimizer=opt)
    print("\n\n#\t tumor model training...\tModel parameters: {0:,}".format(tumor_model.count_params()))

    train_xlist = [row[1] for row in trainingsubset]
    train_ylist = [row[2] for row in trainingsubset]

    valid_xlist = [row[1] for row in validsubset]
    valid_ylist = [row[2] for row in validsubset]

    training_generator   = NpyDataGenerator_Tumor(train_xlist, train_ylist)
    validation_generator = NpyDataGenerator_Tumor(valid_xlist, valid_ylist)
    history_liver = tumor_model.fit_generator( \
                        verbose             = vb,
                        generator           = training_generator,
                        validation_data     = validation_generator,
                        use_multiprocessing = True,
                        workers             = 16,
                        epochs              = settings.options.numepochs,
                        callbacks           = callbacks,
                        shuffle             = True,
                        )
    
    ########
    ########
    ###
    ### TUNE TUMOR MODEL WITH LIVER MODEL MASKS
    ###
    ########
    ########
    saveloclist = settings.options.datafiles_tumor
    loclist = np.genfromtxt(saveloclist, delimiter=',', dtype='str')[1:]
    trainingsubset = [ row for row in loclist if int(row[0]) in train_index]
    testingsubset  = [ row for row in loclist if int(row[0]) in test_index ]
    validsubset    = [ row for row in loclist if int(row[0]) in valid_index]

    opt                 = GetOptimizer(tuning=True)
    callbacks, modelloc = GetCallbacks(logfileoutputdir, "tuning")
    lss, met            = GetLoss('ensemble')
    ensemble            = get_unet_ensemble(liver_model, tumor_model, tune_liver=True, tune_tumor=True)
    ensemble.compile(loss=lss, metrics=met, optimizer=opt)
    ensemble.summary()
    print("\n\n#\t ensemble model on tumor slices...\tModel parameters: {0:,}".format(ensemble.count_params()))
    
    train_xlist = [row[1] for row in trainingsubset]
    train_ylist = [row[2] for row in trainingsubset]

    valid_xlist = [row[1] for row in validsubset]
    valid_ylist = [row[2] for row in validsubset]

    training_generator   = NpyDataGenerator_Ensemble(train_xlist, train_ylist)
    validation_generator = NpyDataGenerator_Ensemble(valid_xlist, valid_ylist)
    history_liver = ensemble.fit_generator( \
                        verbose             = vb,
                        generator           = training_generator,
                        validation_data     = validation_generator,
                        use_multiprocessing = True,
                        workers             = 16,
                        epochs              = settings.options.numepochs//2,
                        callbacks           = callbacks,
                        shuffle             = True,
                        )
    
    '''    
    ########
    ########
    ###
    ### TUNE ENSEMBLE MODEL
    ###
    ########
    ########
    saveloclist = settings.options.datafiles_liver
    loclist = np.genfromtxt(saveloclist, delimiter=',', dtype='str')[1:]
    trainingsubset = [ row for row in loclist if int(row[0]) in train_index]
    testingsubset  = [ row for row in loclist if int(row[0]) in test_index ]
    validsubset    = [ row for row in loclist if int(row[0]) in valid_index]


    opt                 = GetOptimizer(tuning=True)
    callbacks, modelloc = GetCallbacks(logfileoutputdir, "ensemble")
    lss, met            = GetLoss('ensemble')
    for lyr in ensemble.layers:
        lyr.trainable = True
    ensemble.compile(loss=lss, metrics=met, optimizer=opt)
    print("\n\n#\t ensemble model on liver slices...\tModel parameters: {0:,}".format(ensemble.count_params()))
    
    train_xlist = [row[1] for row in trainingsubset]
    train_ylist = [row[2] for row in trainingsubset]

    valid_xlist = [row[1] for row in validsubset]
    valid_ylist = [row[2] for row in validsubset]

    training_generator   = NpyDataGenerator_Ensemble(train_xlist, train_ylist)
    validation_generator = NpyDataGenerator_Ensemble(valid_xlist, valid_ylist)
    history_liver = ensemble.fit_generator( \
                        verbose             = vb,
                        generator           = training_generator,
                        validation_data     = validation_generator,
                        use_multiprocessing = True,
                        workers             = 16,
                        epochs              = settings.options.numepochs//2,
                        callbacks           = callbacks,
                        shuffle             = True,
                        )
    '''
    return modelloc
