import numpy as np
import csv
import sys
import os


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
    if options.liver:
        if not options.datafiles_liver:
            print('no list of liver .npy files given for training')
            quit()
        else:
            saveloclist = options.datafiles_liver
    elif options.tumor:
        if not options.datafiles_tumor:
            print('no list of tumor .npy files given for training')
            quit()
        else:
            saveloclist = options.datafiles_tumor
    else:
        print('not specified liver vs tumor')
        quit()
        
    print('files already generated: using', saveloclist)
    

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

