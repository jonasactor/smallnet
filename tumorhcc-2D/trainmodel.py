import numpy as np
import csv
import sys
import os
import json
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
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
from generator import NpyDataGenerator

###
### Training: build NN model from anonymized data
###
def TrainModel(idfold=0, saveloclist=None):

    from setupmodel import GetSetupKfolds, GetCallbacks, GetOptimizer, GetLoss
    from buildmodel import get_unet
    from generator import setup_training_from_file

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    ###
    ### set up output, logging, and callbacks
    ###
    kfolds = settings.options.kfolds

    logfileoutputdir= '%s/%03d/%03d' % (settings.options.outdir, kfolds, idfold)
    os.system ('mkdir -p ' + logfileoutputdir)
    os.system ('mkdir -p ' + logfileoutputdir + '/nii')
    if settings.options.liver:
        os.system ('mkdir -p ' + logfileoutputdir + '/liver')
    elif settings.options.tumor:
        os.system ('mkdir -p ' + logfileoutputdir + '/tumor')
    else:
        print('need to choose one of {liver,tumor}')
        raise ValueError('need to choose to perform liver or tumor segmentation')
    print("Output to\t", logfileoutputdir)

    ###
    ### load data
    ###

    (train_index,test_index,valid_index) = GetSetupKfolds(settings.options.dbfile, kfolds, idfold)
    if not saveloclist:
        saveloclist = setup_training_from_file()

    loclist = np.genfromtxt(saveloclist, delimiter=',', dtype='str')[1:]
    trainingsubset = [ row for row in loclist if int(row[0]) in train_index]
    testingsubset  = [ row for row in loclist if int(row[0]) in test_index ]
    validsubset    = [ row for row in loclist if int(row[0]) in valid_index]



    ###
    ### create and run model
    ###
    opt                 = GetOptimizer()
    callbacks, modelloc = GetCallbacks(logfileoutputdir, "liver")
    lss, met            = GetLoss()
    model               = get_unet()
    model.summary()
    model.compile(loss  = lss,
        metrics       = met,
        optimizer     = opt)
    print("\n\n\tlivermask training...\tModel parameters: {0:,}".format(model.count_params()))


    train_xlist = [row[1] for row in trainingsubset]
    train_ylist = [row[2] for row in trainingsubset]

    valid_xlist = [row[1] for row in validsubset]
    valid_ylist = [row[2] for row in validsubset]

    training_generator   = NpyDataGenerator(train_xlist, train_ylist)
    validation_generator = NpyDataGenerator(valid_xlist, valid_ylist)
    history_liver = model.fit_generator( \
                        generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=8,
                        # steps_per_epoch  = ntrainslices / settings.options.trainingbatch,
                        epochs           = settings.options.numepochs,
                        callbacks        = callbacks,
                        shuffle          = True,
                        # validation_steps = nvalidslices / settings.options.validationbatch,
                        )
    return modelloc
