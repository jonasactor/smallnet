import numpy as np
import csv
import sys
import os
import json
import keras
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import matplotlib as mptlib
#mptlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

import settings
from setupmodel import GetSetupKfolds
from setupmodel import GetOptimizer, GetLoss
#from buildmodel import get_unet
from mymetrics import dsc_l2, dsc_matlab_l2, dsc_l2_liver, dsc_l2_tumor, dsc_l2_ensemble
#from mymetrics import dsc_l2_3D_npy, dsc_l2_2D_avg_npy, dsc_l2_liver_npy, dsc_l2_tumor_npy
from mymetrics import dsc_l2_3D_npy
from ista import ISTA
from DepthwiseConv3D import DepthwiseConv3D
import preprocess
from generator import NpyDataGenerator_Prediction


def save_nifti(data, name):
   this_nifti = nib.Nifti1Image(data, None)
   this_nifti.to_filename(name)

def PredictNpy(model, imagedata):
    segdata = np.zeros((*imagedata.shape, 1))
    nvalidslices = imagedata.shape[2]

    imagedata   = np.transpose(imagedata, (2,0,1))[...,np.newaxis]
    segdata     = model.predict(imagedata)
    segdata     = np.transpose(segdata,     (1,2,0,3))
    segdata_int = (segdata >= 0.5).astype(settings.SEG_DTYPE)

    return segdata, segdata_int


def PredictNifti(model, saveloc, imageloc, segloc=None):

    print('loading data: ', imageloc)
    image, origheader, trueseg  = preprocess.reorient(imageloc, segloc=segloc)
    image = preprocess.resize_to_nn(image, transpose=False).astype(settings.FLOAT_DTYPE)
    image = preprocess.window(image, settings.options.hu_lb, settings.options.hu_ub)
    image = preprocess.rescale(image, settings.options.hu_lb, settings.options.hu_ub)

    image_img = nib.Nifti1Image(image, None, header=origheader)
    image_img.to_filename(saveloc+'-img.nii')

    predseg_float, predseg = PredictNpy(model, image)

    if segloc:
        trueseg = preprocess.resize_to_nn(trueseg, transpose=False).astype(settings.SEG_DTYPE)
#        names   = ['Background','Liver','Tumor']
#        metrics = [dsc_l2_background_npy, dsc_l2_liver_npy, dsc_l2_tumor_npy]
        names   = ['DSC']
        metrics = [dsc_l2_npy]
        if settings.options.liver:
            tseg = (trueseg >= 1).astype(np.int32)[...,np.newaxis]
        elif settings.options.tumor:
            tseg = (trueseg >  1).astype(np.int32)[...,np.newaxis]

        print('pred ', predseg.shape, predseg.dtype, '\ttrue ',  tseg.shape, tseg.dtype)
        scores  = [ met(tseg.astype(np.int32), predseg.astype(np.float32)) for met in metrics]
        print('DSC:\t', end='')
        for idx,s in enumerate(scores):
            print(names[idx], '\t', 1.0-s, end='\t')
        print()

    print('saving data: ', saveloc)
    print(predseg_float.shape)
    for i in range(predseg_float.shape[-1]):
        segout_float_img = nib.Nifti1Image(predseg_float[...,i], None, header=origheader)
        segout_float_img.to_filename(saveloc+'-float-'+str(i)+'.nii')
        trueseg_i = (trueseg == i).astype(settings.SEG_DTYPE)
        segout_float_img = nib.Nifti1Image(predseg_float[...,i], None, header=origheader)
        segout_float_img.to_filename(saveloc+'-truth-'+str(i)+'.nii')
    trueseg_img = nib.Nifti1Image(trueseg, None, header=origheader)
    trueseg_img.to_filename(saveloc+'-truth.nii')
    predseg_img = nib.Nifti1Image(predseg, None, header=origheader)
    predseg_img.to_filename(saveloc+'-int.nii')

    if settings.options.ttdo and segloc:
        print('starting TTDO...')
        f = K.function([loaded_model.layers[0].input, K.learning_phase()],
                       [loaded_model.layers[-1].output])

        print('\tgenerating trials...')
        results = np.zeros(trueseg.shape + (3,settings.options.ntrials,))
        for jj in range(settings.options.ntrials):
            segdata = np.zeros((*image.shape, 3))
            nvalidslices = image.shape[2]-settings.options.thickness+1
            for z in range(nvalidslices):
                indata = image[np.newaxis,:,:,z:z+settings.options.thickness,np.newaxis]
                segdata[:,:,z:z+settings.options.thickness,:] += f([indata, 1])[0,...]
            for i in range(settings.options.thickness):
                segdata[:,:,i,:] *= (settings.options.thickness)/(i+1)
            for i in range(settings.options.thickness):
                segdata[:,:,-1-i,:] *= (settings.options.thickness)/(i+1)
            results[...,jj] = segdata / settings.options.thickness

        print('\tcalculating statistics...')
        pred_avg = results.mean(axis=-1)
        pred_var = results.var(axis=-1)
        pred_ent = np.zeros(pred_avg.shape)
        ent_idx0 = pred_avg > 0
        ent_idx1 = pred_avg < 1
        ent_idx  = np.logical_and(ent_idx0, ent_idx1)
        pred_ent[ent_idx] = -1*np.multiply(      pred_avg[ent_idx], np.log(      pred_avg[ent_idx])) \
                            -1*np.multiply(1.0 - pred_avg[ent_idx], np.log(1.0 - pred_avg[ent_idx]))

        print('\tsaving statistics...')

        # save pred_avg
        for i in range(pred_avg.shape[-1]):
            segout_float_img = nib.Nifti1Image(pred_avg[...,i], None, header=origheader)
            segout_float_img.to_filename(saveloc+'-avg-float-'+str(i)+'.nii')
        pred_avg_int = np.argmax(pred_avg, axis=-1)
        predseg_img = nib.Nifti1Image(pred_avg_int, None, header=origheader)
        predseg_img.to_filename(saveloc+'-avg-int.nii')

        # save pred_var
        for i in range(pred_var.shape[-1]):
            segout_float_img = nib.Nifti1Image(pred_var[...,i], None, header=origheader)
            segout_float_img.to_filename(saveloc+'-var-float-'+str(i)+'.nii')

        # save pred_ent
        for i in range(pred_ent.shape[-1]):
            segout_float_img = nib.Nifti1Image(pred_ent[...,i], None, header=origheader)
            segout_float_img.to_filename(saveloc+'-ent-float-'+str(i)+'.nii')

    return predseg, names, scores


def PredictKFold(modelloc, dbfile, outdir, kfolds=settings.options.kfolds, idfold=settings.options.idfold, saveloclist=None):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

    (train_index, test_index, valid_index) = GetSetupKfolds(settings.options.dbfile, kfolds, idfold)
    
    logfileoutputdir= '%s/%03d/%03d' % (outdir, kfolds, idfold)
    os.system('mkdir -p '+logfileoutputdir)
    os.system('mkdir -p '+logfileoutputdir + '/predictions')
    savedir = logfileoutputdir+'/predictions'

    print('loading model from', modelloc)
    customdict={ \
            'dsc_l2':            dsc_l2,
            'ISTA':              ISTA,
            'DepthwiseConv3D':   DepthwiseConv3D,
            }
    loaded_model = load_model(modelloc, compile=False, custom_objects=customdict)
    opt          = GetOptimizer()
    lss, met     = GetLoss('ensemble')
    loaded_model.compile(loss=lss, metrics=met, optimizer=opt)
    loaded_model.summary()

    print('loading data from ', saveloclist)
    loclist = np.genfromtxt(saveloclist, delimiter=',', dtype='str')[1:]
    trainingsubset = [row for row in loclist if int(row[0]) in train_index]
    testingsubset  = [row for row in loclist if int(row[0]) in test_index]
    valid_index    = [row for row in loclist if int(row[0]) in valid_index]

    test_xlist = [row[1] for row in testingsubset]
    test_ylist = [row[2] for row in testingsubset]

    for idx in test_index:
        print('\nImage', idx)
        print('gathering image data...')
        this_img_subset = [row for row in loclist if int(row[0]) == idx]
        this_img_xlist  = [row[1] for row in this_img_subset]
        this_img_ylist  = [row[2] for row in this_img_subset]

        img_in = np.empty((len(this_img_xlist), settings.options.trainingresample, settings.options.trainingresample, 1), dtype=settings.FLOAT_DTYPE)
        seg_in = np.empty((len(this_img_ylist), settings.options.trainingresample, settings.options.trainingresample, 1), dtype=settings.SEG_DTYPE)
        cat_in = np.empty((len(this_img_ylist), settings.options.trainingresample, settings.options.trainingresample, 3), dtype=settings.FLOAT_DTYPE)
        for iii in range(len(this_img_xlist)):
            img_in[iii,...] = np.load(this_img_xlist[iii])
            loaded_y        = np.load(this_img_ylist[iii])
            seg_in[iii,...] = loaded_y
            cat_in[iii,...] = to_categorical(loaded_y, num_classes=3)

        print('creating generator and performing prediction...')
        this_img_generator   = NpyDataGenerator_Prediction(this_img_xlist, this_img_ylist, batch_size=8)
        this_img_predictions = loaded_model.predict_generator(this_img_generator)
        this_pred_liver = this_img_predictions[...,0]
        this_pred_tumor = this_img_predictions[...,1]
        this_seg_liver  = (this_pred_liver >= 0.5).astype(settings.SEG_DTYPE)
        this_seg_tumor  = (this_pred_tumor >= 0.5).astype(settings.SEG_DTYPE)

        this_lcc_liver         = preprocess.largest_connected_component(this_seg_liver).astype(settings.SEG_DTYPE)
        this_pred_tumor_masked = this_pred_tumor * this_lcc_liver
        this_seg_tumor_masked  = this_seg_tumor  * this_lcc_liver

        print('saving data to', savedir)

        save_nifti(this_pred_liver,        savedir+'/pred-'+str(idx)+'liver-float.nii')
        save_nifti(this_pred_tumor,        savedir+'/pred-'+str(idx)+'tumor-float.nii')
#        save_nifti(this_seg_liver,         savedir+'/pred-'+str(idx)+'liver-int.nii')
#        save_nifti(this_seg_tumor,         savedir+'/pred-'+str(idx)+'tumor-int.nii')
        save_nifti(this_lcc_liver,         savedir+'/pred-'+str(idx)+'liver-lcc.nii')
#        save_nifti(this_pred_tumor_masked, savedir+'/pred-'+str(idx)+'-tumor-masked-float.nii')
        save_nifti(this_seg_tumor_masked,  savedir+'/pred-'+str(idx)+'-tumor-masked-int.nii')
        save_nifti(img_in[...,0],          savedir+'/img-'+str(idx)+'.nii')
        save_nifti(seg_in[...,0],          savedir+'/seg-'+str(idx)+'.nii')
        save_nifti(cat_in[...,1],          savedir+'/seg-'+str(idx)+'-liver.nii')
        save_nifti(cat_in[...,2],          savedir+'/seg-'+str(idx)+'-tumor.nii')
        

        print('calculating metrics...')
        print('  \t---------------------------------------------------------------------------------')
        print('+ \tDSC-L2 3D LIVER LCC      (int) :\t', dsc_l2_3D_npy(cat_in[...,1,np.newaxis], this_lcc_liver[...,np.newaxis]))
        print('+ \tDSC-L2 3D TUMOR MASKED (float) :\t', dsc_l2_3D_npy(cat_in[...,2,np.newaxis], this_pred_tumor_masked[...,np.newaxis]))
        print('+ \tDSC-L2 3D TUMOR MASKED   (int) :\t', dsc_l2_3D_npy(cat_in[...,2,np.newaxis], this_seg_tumor_masked[...,np.newaxis]))
        print('  \t---------------------------------------------------------------------------------')
        print('+ \tDSC-L2 3D LIVER        (float) :\t', dsc_l2_3D_npy(cat_in[...,1,np.newaxis], this_pred_liver[...,np.newaxis]))
        print('+ \tDSC-L2 3D LIVER          (int) :\t', dsc_l2_3D_npy(cat_in[...,1,np.newaxis], this_seg_liver[...,np.newaxis]))
        print('+ \tDSC-L2 3D TUMOR        (float) :\t', dsc_l2_3D_npy(cat_in[...,2,np.newaxis], this_pred_tumor[...,np.newaxis]))
        print('+ \tDSC-L2 3D TUMOR          (int) :\t', dsc_l2_3D_npy(cat_in[...,2,np.newaxis], this_seg_tumor[...,np.newaxis]))
        print('  \t---------------------------------------------------------------------------------')




def PredictCSV(modelloc, outdir, indir=settings.options.dbfile):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

    logfileoutputdir = outdir
    os.system('mkdir -p '+logfileoutputdir)
    os.system('mkdir -p '+logfileoutputdir + '/predictions')
    savedir = logfileoutputdir+'/predictions'

    print('loading model from', modelloc)
    customdict={ \
            'dsc_l2':            dsc_l2,
            'dsc_matlab':        dsc_matlab,
            'dsc_matlab_l2':     dsc_matlab_l2,
            'dsc_l2_liver':      dsc_l2_liver,
            'dsc_l2_tumor':      dsc_l2_tumor,
            'dsc_l2_background': dsc_l2_background,
            'ISTA':              ISTA,
            'DepthwiseConv3D':   DepthwiseConv3D,
            }
    loaded_model = load_model(modelloc, compile=False, custom_objects=customdict)
    opt          = GetOptimizer()
    lss, met     = GetLoss()
    loaded_model.compile(loss=lss, metrics=met, optimizer=opt)

    with open(indir, 'r') as csvfile:
        myreader = csv.DictReader(csvfile, delimiter=',')
        for row in myreader:
            dataid = int(row['dataid'])
            if dataid in test_index:
                imageloc   = '%s/%s' % (settings.options.rootlocation, row['image'])
                segloc     = '%s/%s' % (settings.options.rootlocation, row['label'])
                saveloc    = savedir+'/pred-'+str(dataid)
                PredictNifti(loaded_model, saveloc, imageloc, segloc=segloc)
