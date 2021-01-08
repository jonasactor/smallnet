import numpy as np
import keras
import keras.backend as K
import tensorflow as tf



###
###
### Similarity scores and metrics
###

def dsc_l2(y_true, y_pred, smooth=0.0000001):
    if len(y_true.shape) == 5:
        num = K.sum(K.square(y_true - y_pred), axis=(1,2,3))
        den = K.sum(K.square(y_true), axis=(1,2,3)) + K.sum(K.square(y_pred), axis=(1,2,3)) + smooth
    elif len(y_true.shape) == 4:
        num = K.sum(K.square(y_true - y_pred), axis=(1,2))
        den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return K.mean(num/den, axis=-1)

def dsc_matlab_l2(y_true, y_pred, smooth=0.0000001):
    if len(y_true.shape) == 5:
        Wk  = K.sum( y_true, axis=(1,2,3)) #dim = (batchsize x n_classes)
        Wk  = Wk*K.square(1./(Wk + 1.))
        num = K.sum( K.square(y_true - y_pred), axis=(1,2,3))
        den = K.sum(K.square(y_true), axis=(1,2,3)) + K.sum(K.square(y_pred), axis=(1,2,3)) + smooth
    elif len(y_true.shape) == 4:
        Wk  = K.sum( y_true, axis=(1,2)) #dim = (batchsize x n_classes)
        Wk  = Wk*K.square(1./(Wk + 1.))
        num = K.sum( K.square(y_true - y_pred), axis=(1,2))
        den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return K.sum( Wk*num) / K.sum(Wk*den)

def dsc_matlab(y_true, y_pred, smooth=0.0000001):
    if len(y_true.shape) == 5:
        Wk  = K.sum( y_true, axis=(1,2,3)) #dim = (batchsize x n_classes)
        Wk  = Wk*K.square(1./(Wk + 1.))
        num = K.sum( y_true*y_pred, axis=(1,2,3)) + smooth
        den = K.sum(K.square(y_true), axis=(1,2,3)) + K.sum(K.square(y_pred), axis=(1,2,3)) + smooth
    elif len(y_true.shape) == 4:
        Wk  = K.sum( y_true, axis=(1,2)) #dim = (batchsize x n_classes)
        Wk  = Wk*K.square(1./(Wk + 1.))
        num = K.sum( y_true*y_pred, axis=(1,2)) + smooth
        den = K.sum(K.square(y_true), axis=(1,2)) + K.sum(K.square(y_pred), axis=(1,2)) + smooth
    return 1.0 - 2*K.sum( Wk*num) / K.sum(Wk*den)


def dsc_l2_background(y_true, y_pred, smooth=0.0000001):
    if len(y_true.shape) == 5:
        y_t = y_true[...,0]
        y_p = y_pred[...,0]
        num = K.sum(K.square(y_t - y_p), axis=(1,2,3))
        den = K.sum(K.square(y_t), axis=(1,2,3)) + K.sum(K.square(y_p), axis=(1,2,3)) + smooth
    elif len(y_true.shape) == 4:
        y_t = y_true[...,0]
        y_p = y_pred[...,0]
        num = K.sum(K.square(y_t - y_p), axis=(1,2)) + smooth
        den = K.sum(K.square(y_t), axis=(1,2)) + K.sum(K.square(y_p), axis=(1,2)) + smooth
    return num/den

def dsc_l2_liver(y_true, y_pred, smooth=0.0000001):
    if len(y_true.shape) == 5:
        y_t = y_true[...,1]
        y_p = y_pred[...,1]
        num = K.sum(K.square(y_t - y_p), axis=(1,2,3))
        den = K.sum(K.square(y_t), axis=(1,2,3)) + K.sum(K.square(y_p), axis=(1,2,3)) + smooth
    elif len(y_true.shape) == 4:
        y_t = y_true[...,1]
        y_p = y_pred[...,1]
        num = K.sum(K.square(y_t - y_p), axis=(1,2)) + smooth
        den = K.sum(K.square(y_t), axis=(1,2)) + K.sum(K.square(y_p), axis=(1,2)) + smooth
    return num/den

def dsc_l2_tumor(y_true, y_pred, smooth=0.0000001):
    if len(y_true.shape) == 5:
        y_t = y_true[...,2]
        y_p = y_pred[...,2]
        num = K.sum(K.square(y_t - y_p), axis=(1,2,3)) + smooth
        den = K.sum(K.square(y_t), axis=(1,2,3)) + K.sum(K.square(y_p), axis=(1,2,3)) + smooth
    elif len(y_true.shape) == 4:
        y_t = y_true[...,2]
        y_p = y_pred[...,2]
        num = K.sum(K.square(y_t - y_p), axis=(1,2)) + smooth
        den = K.sum(K.square(y_t), axis=(1,2)) + K.sum(K.square(y_p), axis=(1,2)) + smooth
    return num/den



###
### npy versions
###

def dsc_l2_npy(y_true, y_pred, smooth=0.0000001):
    num = np.sum(np.square(y_true - y_pred))
    den = np.sum(np.square(y_true)) + np.sum(np.square(y_pred)) + smooth
    return num/den

def dsc_l2_background_npy(y_true, y_pred, smooth=0.0000001):
    y_t = (y_true == 0).astype(np.int32)
    y_p = (y_pred == 0).astype(np.int32)
    num = np.sum(np.square(y_t - y_p))
    den = np.sum(np.square(y_t)) + np.sum(np.square(y_p)) + smooth
    return num/den

def dsc_l2_liver_npy(y_true, y_pred, smooth=0.0000001):
    y_t = (y_true == 1).astype(np.int32)
    y_p = (y_pred == 1).astype(np.int32)
    num = np.sum(np.square(y_t - y_p))
    den = np.sum(np.square(y_t)) + np.sum(np.square(y_p)) + smooth
    return num/den

def dsc_l2_tumor_npy(y_true, y_pred, smooth=0.0000001):
    y_t = (y_true == 2).astype(np.int32)
    y_p = (y_pred == 2).astype(np.int32)
    num = np.sum(np.square(y_t - y_p))
    den = np.sum(np.square(y_t)) + np.sum(np.square(y_p)) + smooth
    return num/den
