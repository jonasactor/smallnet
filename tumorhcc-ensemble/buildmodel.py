import numpy as np
import keras
from keras.layers import Conv3D, Add, SpatialDropout3D, AveragePooling3D, UpSampling3D, Input, MaxPooling3D, Concatenate, Activation, Multiply, Lambda
from keras.layers import Conv2D, SpatialDropout2D, AveragePooling2D, UpSampling2D, MaxPooling2D, DepthwiseConv2D, Conv2DTranspose, LocallyConnected2D
from DepthwiseConv3D import DepthwiseConv3D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1, l2
import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf

import settings
from ista import ISTA


def ConvBlock(model_in, filters=settings.options.filters, add=True, drop=True, use_depthwise=settings.options.depthwise):
    kreg = None
    wreg = None
    if settings.options.l1reg:
        kreg=l1(settings.options.l1reg)

    if use_depthwise:
        model = DepthwiseConv2D( \
            kernel_size=(5,5),
            padding='same',
            depth_multiplier=4,
            activation='linear',
            use_bias=False,
            kernel_regularizer=kreg )(model_in)
        model = Conv2D( \
            filters=filters,
            kernel_size=(1,1),
            padding='same',
            activation=settings.options.activation,
            kernel_regularizer=wreg,
            use_bias=True)(model)
    else:
        model = Conv2D( \
            filters=filters,
            kernel_size=(3,3),
            padding='same',
            activation=settings.options.activation,
            kernel_regularizer=kreg,
            use_bias=True)(model_in)
    if drop:
        model = SpatialDropout2D(settings.options.dropout)(model)
    if add:
        model = Add()([model_in, model])
    return model


def Block(model, filters):
    if settings.options.densenet:
        model1 = ConvBlock(model,  add=False, drop=False, filters=filters)
        model2 = ConvBlock(model1, add=False, drop=False, filters=filters)
        model  = Concatenate()([model, model1, model2])
        model  = Conv2D( \
                filters=filters,
                kernel_size=(1,1),
                padding='same',
                activation=settings.options.activation)(model)
    elif settings.options.unet:
        model = ConvBlock(model, add=False, drop=False, filters=filters)
        model = ConvBlock(model, add=False, drop=False, filters=filters)
    elif settings.options.resnet:
        model = ConvBlock(model, add=True, drop=False, filters=filters)
        model = ConvBlock(model, add=True, drop=False, filters=filters)
    return model


def module_mid(model, depth, filters=settings.options.filters):
    if depth==0:
        return Block(model, filters)

    else:

        m_down = Block(model, filters=filters)
        
        if not settings.options.pocket:
            filters*=2
            m_down = Conv2D( \
                filters=filters,
                kernel_size=(1,1),
                padding='same',
                activation=None)(m_down)
        m_mid = MaxPooling2D()(m_down)
        
        m_mid = module_mid(m_mid, depth=depth-1, filters=filters)
    
        if not settings.options.pocket:
            filters = int(filters/2)

        if settings.options.conv2Dtranspose:
            m_up = Conv2DTranspose( \
                    filters=filters,
                    kernel_size=(2,2),
                    padding='same',
                    activation=None,
                    strides=(2,2)   )(m_mid)
        else:
            m_up = UpSampling2D()(m_mid)
            if settings.options.pocket:
                m_up = Add()([m_up, m_down])
            else:
                m_up = Concatenate()([m_up, m_down])
                m_up = Conv2D( \
                    filters=filters,
                    kernel_size=(1,1),
                    padding='same',
                    activation=None)(m_up)

        m_up = Block(m_up, filters=filters)
        return m_up


def unet(layer_in, depth=settings.options.depth):
    return module_mid(layer_in, depth=depth)

def CapLayer(layer_in, act_f=None, classes=1):
    layer = SpatialDropout2D(settings.options.dropout)(layer_in)
    layer = Conv2D(\
            filters=classes,
            kernel_size=(1,1),
            padding='same',
            activation=act_f,
            use_bias=True)(layer)
    return layer


def get_unet_liver():
    _filters  = settings.options.filters
    indim = (settings._ny, settings._nx)
    img_in = Input(shape=(*indim, 1))

    _features = Conv2D(\
            filters=_filters,
            kernel_size=(7,7),
            padding='same',
            activation=settings.options.activation)(img_in)
    _unet = unet(_features)
    _unet = Add()([_features, _unet])
    _out = CapLayer(_unet, classes=1, act_f='sigmoid')
    _model = Model(inputs=img_in, outputs=_out)
    _model.summary()
    if settings.options.gpu > 1:
        return  multi_gpu_model(_model, gpus=settings.options.gpu)
    return _model

def get_unet_tumor(liver_model):
    _filters = settings.options.filters
    indim = (settings._ny, settings._nx)

    all_in = Input(shape=(*indim, 2))

    img = Lambda( lambda x: x[...,0,np.newaxis] )(all_in)
    label = Lambda( lambda x: x[...,1,np.newaxis])(all_in)
    mask = Multiply()([img, label])
    all_in3 = Concatenate()([img, label, mask])

    tumor_features = Conv2D(\
            filters=_filters,
            kernel_size=(7,7),
            padding='same',
            activation=settings.options.activation)(all_in3)
    tumor_unet = unet(tumor_features)
    tumor_out  = Add()([tumor_features, tumor_unet])
    tumor_out  = CapLayer(tumor_out, classes=1, act_f='sigmoid')
    tumor_model = Model(inputs=all_in, outputs=tumor_out)
    tumor_model.summary()
#    print('\t preloading weights from liver model...')
#    for i, lyr in enumerate(liver_model.layers):
#        if lyr.name == 'conv2d_1':
#            print(lyr)
#        else:
#             tumor_model.layers[i-4].set_weights(lyr.get_weights())

    if settings.options.gpu > 1:
        return multi_gpu_model(tumor_model, gpus=settings.options.gpu)
    return tumor_model
    
def get_unet_ensemble(liver_model, tumor_model, tune_liver=False, tune_tumor=True):
    _filters = settings.options.filters
    indim = (settings._ny, settings._nx)

    img_in = Input(shape=(*indim, 1))

    liver_model.trainable=tune_liver
    liver_seg = liver_model(img_in)

    tumor_model.trainable=tune_tumor
    tumor_ins = Concatenate()([img_in, liver_seg])
    tumor_seg = tumor_model(tumor_ins)

    two_segs = Concatenate()([liver_seg, tumor_seg])
    model = Model(inputs=img_in, outputs=two_segs)
    if settings.options.gpu > 1:
        model = multi_gpu_model(model, gpus=settings.options.gpu)

    return model

