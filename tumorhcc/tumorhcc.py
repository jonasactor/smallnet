import numpy as np
import csv
import sys
import os
# import json
# import keras
# from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D
# from keras.models import Model, Sequential
# from keras.models import model_from_json, load_model
# from keras.utils import multi_gpu_model
# from keras.utils.np_utils import to_categorical
# import keras.backend as K
# from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
# from keras.callbacks import Callback as CallbackBase
# from keras.preprocessing.image import ImageDataGenerator
# from keras.initializers import Constant
# import nibabel as nib
# from scipy import ndimage
# from sklearn.model_selection import KFold
# import skimage.transform
# import matplotlib as mptlib
# #mptlib.use('TkAgg')
# import matplotlib.pyplot as plt
# import tensorflow as tf


import settings
from settings import process_options, perform_setup
(options, args) = process_options()

IMG_DTYPE, SEG_DTYPE, _nx, _ny = perform_setup(options)

# from setupmodel import GetDataDictionary, BuildDB
from trainmodel import TrainModel
from predictmodel import PredictCSV, PredictNifti
from kfolds import OneKfold, Kfold
from generator import *

if ( (not options.trainmodel) and (not options.predictmodel) ):
    print("parser error")
    quit()

if options.trainmodel:
    if not options.datafiles:
        print('starting data file generation')
        saveloclist = setup_training_from_file()
    else:
        print('files already generated: using', options.datafiles)
        saveloclist = options.datafiles

    if options.kfolds > 1:
        if options.idfold > -1:
            OneKfold(i=options.idfold, saveloclist=saveloclist)
        else:
            Kfold(saveloclist=saveloclist)
    else:
        TrainModel(saveloclist=saveloclist)

if options.predictmodel:
    if options.predictfromcsv:
        PredictCSV(modelloc=options.predictmodel, outdir=options.outdir, indir=options.predictfromcsv)
    else:
        PredictNifti(model, options.outdir+'/predictions/pred', options.predictimage, segloc=None)

    PredictModel()
