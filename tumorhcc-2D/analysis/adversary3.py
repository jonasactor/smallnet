import numpy as np
import os
from optparse import OptionParser
import matplotlib as mptlib
mptlib.use('TkAgg')
import matplotlib.pyplot as plt

import nibabel as nib

from scipy import ndimage
import skimage.transform
from skimage.measure import label

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
sys.path.append('/rsrch1/ip/jacctor/smallnet/tumorhcc-2D')
from mymetrics import dsc_l2
import preprocess
import settings

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8
FLOAT_DTYPE = np.float32

_globalexpectedpixel=512
_nx = 256
_ny = 256

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
parser.add_option( "--imgloc",
        action="store", dest="imgloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/volume-124.nii',
        help="img location", metavar="PATH.nii")
parser.add_option( "--segloc",
        action="store", dest="segloc", default='/rsrch1/ip/jacctor/LiTS/LiTS/TrainingBatch2/segmentation-124.nii',
        help="seg location", metavar="PATH.nii")
parser.add_option( "--delta",
        type="float", dest="delta", default=0.1,
        help="perturbation", metavar="float")
(options, args) = parser.parse_args()



# cut off img intensities
# img : npy array
# lb  : desired lower bound
# ub  : desired upper bound
def window(img, lb, ub):
    too_low  = img <= lb
    too_high = img >= ub
    img[too_low]  = lb
    img[too_high] = ub
    return img

# rescale img to [-1,   1] if no augmentation
# rescale img to [ 0, 255] if augmentation
# img : npy array
# lb  : known lower bound in image
# ub  : known upper bound in image
def rescale(img, lb, ub):
    rs = img.astype(FLOAT_DTYPE)
    rs = 2.*(rs - lb)/(ub-lb) - 1.0
    rs = rs.astype(FLOAT_DTYPE)
    return rs

# reorient NIFTI files into RAS+
# takes care to perform same reorientation for both image and segmentation
# takes image header as truth if segmentation and image headers differ
def reorient(imgloc, segloc=None):

    imagedata   = nib.load(imgloc)
    orig_affine = imagedata.affine
    orig_header = imagedata.header
    imagedata   = nib.as_closest_canonical(imagedata)
    img_affine  = imagedata.affine
    numpyimage = imagedata.get_data().astype(IMG_DTYPE)
    numpyseg   = None
    print('image :    ', nib.orientations.aff2axcodes(orig_affine), ' to ', nib.orientations.aff2axcodes(img_affine))

    if segloc is not None:
        segdata    = nib.load(segloc)
        old_affine = segdata.affine
        segdata    = nib.as_closest_canonical(segdata)
        seg_affine = segdata.affine
        if not np.allclose(seg_affine, img_affine):
            segcopy = nib.load(segloc).get_data()
            copy_header = orig_header.copy()
            segdata = nib.nifti1.Nifti1Image(segcopy, orig_affine, header=copy_header)
            segdata = nib.as_closest_canonical(segdata)
            seg_affine = segdata.affine
        print('seg   :    ', nib.orientations.aff2axcodes(old_affine), ' to ', nib.orientations.aff2axcodes(seg_affine))
        numpyseg = segdata.get_data().astype(SEG_DTYPE)

    return numpyimage, orig_header, numpyseg

def get_num_slices(imgloc):
    imagedata = nib.load(imgloc)
    orig_header = imagedata.header
    imageshape = orig_header.get_data_shape()
    print(imageshape)
    return imageshape

# sample down to nn's expected input size
def resize_to_nn(img,transpose=True):
    if img.shape[1] == _nx  and img.shape[0] == _ny:
        expected = img
    else:
        expected = skimage.transform.resize(img,
            (_nx,_ny,img.shape[2]),
            order=0,
            mode='constant',
            preserve_range=True)
    if transpose:
        expected = expected.transpose(2,1,0)
    return expected

# return to original size
def resize_to_original(img,transpose=True,dtype=np.float32):
    real = skimage.transform.resize(img,
            (img.shape[0],_globalexpectedpixel,_globalexpectedpixel),
            order=0,
            mode='constant',
            preserve_range=True,
            dtype=dtype)
    if transpose:
        real = real.transpose(2,1,0)
    return real

# returns largest connected component of {0,1} binary segmentation image
# in : img \in {0,1}^n_pixels
def largest_connected_component(img):
    labels = label(img)
    assert ( labels.max() != 0)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC





###
### start a session - will need same session to link K to tf
###

session = tf.Session()
K.set_session(session)

###
### get data
###

def get_img(imgloc, segloc=None):
    npimg, _, npseg = reorient(imgloc, segloc)
    npimg           = resize_to_nn(npimg).astype(np.float32)
    npimg           = window(npimg, -100,200)
    npimg           = rescale(npimg, -100,200)
    npseg           = resize_to_nn(npseg).astype(np.uint8)

    print(npimg.shape)
    print(npseg.shape)
    assert npimg.shape == npseg.shape

    return npimg[...,np.newaxis], npseg[...,np.newaxis]
###
### WARNING: can't return only one slice for evaluation if there are 2 gpus, since it will try 
###          to send one slice to each gpu and then not send anything to gpu:1
###          otherwise error: check failed work_element_count > 0 (0 vs 0)
#    midslice = npimg[int(npimg.shape[0] / 2),:,:]
#    midseg   = npseg[int(npimg.shape[0] / 2),:,:]
#    return midslice[np.newaxis,:,:,np.newaxis], midseg[np.newaxis,:,:,np.newaxis]

x_test, y_test = get_img(options.imgloc, options.segloc)



###
### load model and perform adversarial attack
###

print(options.model)
net = load_model(options.model, custom_objects={'dsc_l2':dsc_l2})

def l2loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true,y_pred)

def fgsm(ximg, yseg, net, loss=l2loss, eps=options.delta):

    print(ximg.shape)
    print(yseg.shape)

    x_adv   = np.zeros_like(ximg)
    perturb = np.zeros_like(ximg)
    x_grad  = np.zeros_like(ximg)

    nslices = ximg.shape[0]
    sss = [32*s for s in range(nslices//32)]
    if nslices%32 > 1:
        sss.append(nslices)
    print(sss)
    for s in range(len(sss)-1):

        srt = sss[s]
        end = sss[s+1]
        x = ximg[srt:end,...]
        y = yseg[srt:end,...]
        #x = x[np.newaxis,...]
        #y = y[np.newaxis,...]

        yy = y.astype(np.float32)
        loss_vec = loss(yy, net.output)
        grad = K.gradients(loss_vec, net.input)
        gradfunc = K.function([net.input], grad)
        grad_at_x = gradfunc([x])
        perturb_this = eps*np.sign(grad_at_x[0])
        x_adv_this = x + perturb_this
        x_grad_this = grad_at_x[0]

        x_adv[srt:end,...] = x_adv_this
        perturb[srt:end,...] = perturb_this
        x_grad[srt:end,...] = x_grad_this

    return x_adv, perturb, x_grad


losses = ['l2', 'dsc']
lossfxns = [l2loss, dsc_l2]

for iii in range(len(losses)):

    lll = losses[iii]
    lfn = lossfxns[iii]

    x_adv, perturb, x_grad = fgsm(x_test, y_test, net, eps=options.delta, loss=lfn)

    x_adv_orig = x_adv[...,0]
    x_adv_orig = np.transpose(x_adv_orig, (1,2,0))
    x_adv_orig_nii = nib.Nifti1Image(x_adv_orig , None )
    x_adv_orig_nii.to_filename(options.outdir+lll+'/'+'img-adv.nii.gz')

    perturb_orig = perturb[...,0]
    perturb_orig = np.transpose(perturb_orig, (1,2,0))
    perturb_orig_nii = nib.Nifti1Image(perturb_orig, None )
    perturb_orig_nii.to_filename(options.outdir+lll+'/'+'perturb.nii.gz')

    x_grad_orig = x_grad[...,0]
    x_grad_orig = np.transpose(x_grad_orig, (1,2,0))
    x_grad_orig_nii = nib.Nifti1Image(x_grad_orig, None)
    x_grad_orig_nii.to_filename(options.outdir+lll+'/'+'grad.nii.gz')

    x_test_orig = x_test[...,0]
    x_test_orig = np.transpose(x_test_orig, (1,2,0))
    x_test_orig_nii = nib.Nifti1Image(x_test_orig , None )
    x_test_orig_nii.to_filename(options.outdir+lll+'/'+'img.nii.gz')

    y_test_orig = y_test[...,0]
    y_test_orig = np.transpose(y_test_orig, (1,2,0))
    y_test_orig_nii = nib.Nifti1Image(y_test_orig , None )
    y_test_orig_nii.to_filename(options.outdir+lll+'/'+'seg.nii.gz')

    y_pred = net.predict(x_test, batch_size=16)
    y_pred_orig = y_pred[...,0]
    y_pred_orig = np.transpose(y_pred_orig, (1,2,0))
    y_pred_orig_nii = nib.Nifti1Image(y_pred_orig, None )
    y_pred_orig_nii.to_filename(options.outdir+lll+'/'+'seg-pred.nii.gz')

    y_adv = net.predict(x_adv, batch_size=16)
    y_adv_orig = y_adv[...,0]
    y_adv_orig = np.transpose(y_adv_orig, (1,2,0))
    y_adv_orig_nii = nib.Nifti1Image(y_adv_orig , None )
    y_adv_orig_nii.to_filename(options.outdir+lll+'/'+'seg-pred-adv.nii.gz')



