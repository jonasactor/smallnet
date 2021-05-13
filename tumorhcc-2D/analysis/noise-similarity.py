import os
import csv
import time
import json
from optparse import OptionParser

import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib
import skimage.transform

import keras
import keras.backend as K
import keras.losses
from keras import models
from keras import layers
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import sys
sys.setrecursionlimit(5000)
sys.path.append('/rsrch1/ip/jacctor/livermask/liverhcc')
from mymetrics import dsc_l2, dsc, dsc_int, l1, dsc_int_3D, dsc_l2_3D
from ista import ISTA
import preprocess

import math
import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.signal import convolve2d

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

_globalexpectedpixel=512
_nx = 256
_ny = 256

npx = 256
hu_lb = -100
hu_ub = 400 
std_lb = 0
std_ub = 100



###
### set options
###
parser = OptionParser()
parser.add_option( "--model",
        action="store", dest="model", default=None,
        help="model location", metavar="PATH")
parser.add_option( "--outdir",
        action="store", dest="outdir", default="./",
        help="out location", metavar="PATH")
parser.add_option( "--saveloc",
        action="store", dest="saveloc", default=None,
        help="location to save images", metavar="PATH")
parser.add_option( "--imgloc",
        action="store", dest="imgloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-124.nii',
        help="img location", metavar="PATH.nii")
parser.add_option( "--segloc",
        action="store", dest="segloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-124.nii',
        help="seg location", metavar="PATH.nii")
parser.add_option( "--delta",
        type="float", dest="delta", default=0.1,
        help="perturbation", metavar="float")
parser.add_option( "--show",
        action="store_true", dest="show", default=False,
        help="show plots", metavar="bool")
parser.add_option( "--test",
        action="store_true", dest="test", default=False,
        help="small sets for testing purposes", metavar="bool")
parser.add_option( "--one_at_a_time",
        action="store_true", dest="one_at_a_time", default=False,
        help="process one image at a time, using a bash script instead of a loop : saves on memory", metavar="bool")
parser.add_option( "--idxOne",
        type="int", dest="idxOne", default=124,
        help="choice for command line image idx", metavar="int")
(options, args) = parser.parse_args()



rootloc_mda = '/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse'
if options.test:
    dbfile_mda = '/rsrch1/ip/jacctor/livermask/trainingdata-mda-small.csv'
else:
    dbfile_mda = '/rsrch1/ip/dtfuentes/github/RandomForestHCCResponse/datalocation/trainingdata.csv'
#    dbfile_mda = '/rsrch1/ip/jacctor/livermask/trainingdata-mda-small.csv'


###
### start a session - will need same session to link K to tf
###

session = tf.Session()
K.set_session(session)

###
### get data
###

def get_img(imgloc, segloc=None):
    npimg, _, npseg = preprocess.reorient(imgloc, segloc)
    npimg           = preprocess.resize_to_nn(npimg).astype(np.float32)
    npimg           = preprocess.window(npimg,  hu_lb, hu_ub)
    npimg           = preprocess.rescale(npimg, hu_lb, hu_ub)
    npseg           = preprocess.resize_to_nn(npseg).astype(np.uint8)

    print(npimg.shape)
    assert npimg.shape == npseg.shape
    return npimg, npseg
###
### WARNING: can't return only one slice for evaluation if there are 2 gpus, since it will try 
###          to send one slice to each gpu and then not send anything to gpu:1
###          otherwise error: check failed work_element_count > 0 (0 vs 0)
#    midslice = npimg[int(npimg.shape[0] / 2),:,:]
#    midseg   = npseg[int(npimg.shape[0] / 2),:,:]
#    return midslice[np.newaxis,:,:,np.newaxis], midseg[np.newaxis,:,:,np.newaxis]




def save_as_nii(img, svloc):
    if len(img.shape) == 4:
        to_sv = img[...,0]
    elif len(img.shape) == 3:
        to_sv = img
    else:
        print('failed to save : dimensions not clear')
        return
    to_sv_nii = nib.Nifti1Image(to_sv, None)
    to_sv_nii.to_filename(svloc)


###
###
### NOISE CONSTRUCTORS
###
### The following functions take data + parameters and
### and return a noisy image, with noise following the
### following options:
###     1. gaussian noise
###     2. uniform noise
###     3. sparse gaussian noise
###     4. sparse uniform noise
###     5. adversarial uniform noise
###     6. simulated physical noise (avg)
###     7. salt and pepper noise
###
###



##########
##########
### 1. ###
##########
##########

# noise follows N(0, eps)
def make_gaussian_noise(data, eps=0.01):
    noise = np.random.normal(0, eps, data.shape)
    return noise 


##########
##########
### 5. ###
##########
##########

# adversarial attack via FGSM
# (i.e. uniform-noise attack)
def fgsm(ximg, yseg, net, loss=dsc_l2, eps=options.delta):


    ximg = ximg[...,np.newaxis]
    yseg = yseg[...,np.newaxis]
    perturb_adv    = np.zeros_like(ximg)

    # need to batch, otherwise gradient gets too big to fit in memory
    nslices = ximg.shape[0]
    sss = [32*s for s in range(nslices//32)]
    if nslices%32 > 1:
        sss.append(nslices)

    for s in range(len(sss)-1):

        srt = sss[s]
        end = sss[s+1]
        x = ximg[srt:end,...]
        y = yseg[srt:end,...]

        yy = y.astype(np.float32)
        loss_vec = loss(yy, net.output)
        grad = K.gradients(loss_vec, net.input)
        gradfunc = K.function([net.input], grad)
        grad_at_x = gradfunc([x])

        perturb_this = np.clip(grad_at_x[0], -1.0, 1.0)
        perturb_this[np.abs(perturb_this) < 0.0001] = 0.0
#        perturb_this = np.sign(grad_at_x[0])
        perturb_adv[srt:end,...] = perturb_this

    return perturb_adv[...,0]
    return preprocess.resize_to_original(perturb_adv[...,0]).astype(np.float32)



##########
##########
### 6. ###
##########
##########

# simulated physical noise

# calculate local empirical noise on 2d slices
def get_noise_dist_2d(data, k=5):
    ker = np.ones((k,k))/(k**2) 
    mean = convolve2d(data,                 ker, mode='same', boundary='fill', fillvalue=0)
    stdv = convolve2d(np.square(data-mean), ker, mode='same', boundary='fill', fillvalue=0)
    return stdv * np.sign(data - mean)

# calculate local empirical noise on 3d stack
# calculations are done 2d-slicewise to deal with anisotropic voxels
def get_noise_dist_3d(data3d, k=5):
    error = np.zeros_like(data3d)
    nslices = data3d.shape[0]
    for s in range(nslices):
        error[s,...] = get_noise_dist_2d(data3d[s,...], k=k)
    error[np.abs(error) < 0.0001] = 0.0
    return error




def compute_similarity(imgloc, segloc, loaded_net, im_idx):
    
    img, seg = get_img(imgloc, segloc)

    print('\t generating adversary...')
    per_adv = fgsm(img, seg, loaded_net, eps=1.0, loss=dsc_l2)

    counts_adv   = np.count_nonzero(per_adv, axis=(1,2))
    counts_liver = np.count_nonzero(seg,     axis=(1,2))
    idx_adv      = counts_adv   > 0
    idx_liver    = counts_liver > 0
    sl           = np.argmax(counts_liver)

    print('\t generating local noise...')
    per_err = get_noise_dist_3d(img, k=5)
    per_err[np.abs(per_err) > 0.025] = 0.0

    print('\t computing similarity...')
    cc1 = np.zeros(img.shape[0])
    cc2 = np.zeros(img.shape[0])
    cc3 = np.zeros(img.shape[0])
    slist = [s for s in range(img.shape[0])]
    for s in range(img.shape[0]):
        cc1[s] = sp.spatial.distance.cosine(per_adv[s,...].flatten(), per_err[s,...].flatten())
        cc2[s] = sp.spatial.distance.cosine(np.sign(per_adv[s,...]).flatten(), np.sign(per_err[s,...]).flatten())
        cc3[s] = sp.spatial.distance.cosine(np.abs(per_adv[s,...]).flatten(), np.abs(per_err[s,...]).flatten())
    if options.show:
        plt.scatter(slist, cc1, c='b')
        plt.scatter(slist, cc2, c='r')
        plt.scatter(slist, cc3, c='g')
        plt.show()

    cos1 = sp.spatial.distance.cosine(per_adv.flatten(), per_err.flatten())
    cos2 = sp.spatial.distance.cosine(np.sign(per_adv).flatten(), np.sign(per_err).flatten())
    cos3 = sp.spatial.distance.cosine(np.abs(per_adv).flatten(), np.abs(per_err).flatten())
    print('similarity          : ', cos1, np.mean(cc1[idx_adv]))
    print('signed similarity   : ', cos2, np.mean(cc2[idx_adv]))
    print('absval similarity   : ', cos3, np.mean(cc3[idx_adv]))
    print('largest slice index : ', sl)

    if options.show:
#        plt.figure(figsize=(1,1))
        plt.subplot(2,2,1)
        plt.imshow(img[sl,...], cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,2)
        plt.imshow((seg[sl,...] > 0).astype(np.float32), cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,3)
        plt.imshow(per_adv[sl,...], cmap='gray')
        plt.axis('off')
        plt.subplot(2,2,4)
        plt.imshow(per_err[sl,...], cmap='gray')
        plt.axis('off')

        plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0, wspace=0.1, hspace=0.1)
        plt.show()

    if options.saveloc is not None:
        plt.imshow(img[sl,...], cmap='gray')
        plt.axis('off')
        plt.savefig(options.saveloc+'/'+'noisecomp-'+str(im_idx)+'-img.png', bbox_inches="tight")
        plt.close()

        plt.imshow((seg[sl,...] > 0).astype(np.float32), cmap='gray')
        plt.axis('off')
        plt.savefig(options.saveloc+'/'+'noisecomp-'+str(im_idx)+'-seg.png', bbox_inches="tight")
        plt.close()

        plt.imshow(per_adv[sl,...], cmap='gray')
        plt.axis('off')
        plt.savefig(options.saveloc+'/'+'noisecomp-'+str(im_idx)+'-adv.png', bbox_inches="tight")
        plt.close()
        
        plt.imshow(per_err[sl,...], cmap='gray')
        plt.axis('off')
        plt.savefig(options.saveloc+'/'+'noisecomp-'+str(im_idx)+'-err.png', bbox_inches="tight")
        plt.close()


    del img
    del seg
    del per_adv
    del per_err
    del counts_adv
    del counts_liver
    del idx_liver
    del cc1
    del cc2
    return cc3[idx_adv].tolist()


if options.test:
    locdict =    { \
        124: { \
            'vol': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-124.nii',
            'seg': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-124.nii' },
        130: {  \
           'vol': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-130.nii',
           'seg': '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-130.nii' }, }
elif options.one_at_a_time:
    locdict = {}
    base_vol_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-'
    base_seg_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-'
    i = options.idxOne
    locdict[i] = { 'vol' : base_vol_string+str(i)+'.nii', 'seg': base_seg_string+str(i)+'.nii' }
else:
    locdict = {}
    base_vol_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-'
    base_seg_string = '/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-'
    for i in range(111, 131):
        locdict[i] = { 'vol' : base_vol_string+str(i)+'.nii', 'seg': base_seg_string+str(i)+'.nii' }



net_train = load_model(options.model, custom_objects={'dsc_l2':dsc_l2, 'dsc':dsc, 'dsc_int':dsc_int, 'ISTA':ISTA})
cclist = []
for im_idx in locdict:

    print(im_idx)
    imgloc = locdict[im_idx]['vol']
    segloc = locdict[im_idx]['seg']
    nm = compute_similarity(imgloc, segloc, net_train, im_idx)
    cclist += nm
    print('\n\n')

ccarray = np.asarray(cclist)
print('absolute similarity avg across all', len(cclist), ' slices is', np.mean(ccarray))

plt.hist(cclist, bins=100, range=(0,1))
plt.show()

print('\ndone.')




