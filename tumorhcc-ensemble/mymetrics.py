import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy


###
###
### Similarity scores and metrics
###

def dsc_l2(y_true, y_pred, smooth=0.0000001):
    num = K.sum(K.square(y_true - y_pred), axis=(1,2))
    den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return K.mean(num/den, axis=-1)

def dsc_matlab_l2(y_true, y_pred, smooth=0.0000001):
    Wk  = K.sum( y_true, axis=(1,2)) #dim = (batchsize x n_classes)
    Wk  = Wk*K.square(1./(Wk + 1.))
    num = K.sum( K.square(y_true - y_pred), axis=(1,2))
    den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return K.sum( Wk*num) / K.sum(Wk*den)

def dsc_l2_ensemble(y_true, y_pred, smooth=0.0001):
    ylt = y_true[...,0]
    ytt = y_true[...,1]
    ylp = y_pred[...,0]
    ytp = y_pred[...,1]
    return 0.5*(dsc_l2(ylt, ylp) + dsc_l2(ytt, ytp))

def dsc_l2_liver(y_true, y_pred, smooth=0.0001):
    ylt = y_true[...,0]
    ylp = y_pred[...,0]
    return dsc_l2(ylt, ylp)

def dsc_l2_tumor(y_true, y_pred, smooth=0.0001):
    ytt = y_true[...,1]
    ytp = y_pred[...,1]
    return dsc_l2(ytt, ytp)

def dsc_l2_bxe(y_true, y_pred, smooth=0.0001):
    return dsc_l2(y_true, y_pred, smooth=smooth) + K.sum(binary_crossentropy(y_true, y_pred), axis=(1,2))


###
### npy versions
###

def dsc_l2_3D_npy(y_true, y_pred, smooth=0.0000001):
    num = np.sum(np.square(y_true - y_pred))
    den = np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth
    return num/den

'''
def dsc_l2_2D_avg_npy(y_true, y_pred, smooth=0.0000001):
    num = np.sum(np.square(y_true - y_pred), axis=(1,2,3))
    den = np.sum(np.square(y_true), axis=(1,2,3)) + np.sum(np.square(y_pred), axis=(1,2,3)) + smooth
    return np.average( num / den )

def dsc_l2_liver_npy(y_true, y_pred, smooth=0.00000001):
    y_t = y_true[...,1]
    y_p = y_pred[...,1]
    num = np.sum(np.square(y_t - y_p), axis=(1,2))
    den = np.sum(np.square(y_t), axis=(1,2)) + np.sum(np.square(y_p), axis=(1,2)) + smooth
    return np.average( num / den )

def dsc_l2_tumor_npy(y_true, y_pred, smooth=0.0000001):
    y_t = y_true[...,2]
    y_p = y_pred[...,2]
    num = np.sum(np.square(y_t - y_p), axis=(1,2))
    den = np.sum(np.square(y_t), axis=(1,2)) + np.sum(np.square(y_p), axis=(1,2)) + smooth
    return np.average( num / den )
'''
