import numpy as np
import keras
from keras.layers import Conv3D, Add, SpatialDropout3D, AveragePooling3D, UpSampling3D, Input, MaxPooling3D
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


def Block(model_in, filters=settings.options.filters, add=True, drop=False):
    kreg = None
    wreg = None
    if settings.options.l1reg:
        kreg=l1(settings.options.l1reg)
#    model = DepthwiseConv3D( \
#        kernel_size=(3,3,3),
#        padding='same',
#        depth_multiplier=1,
#        activation=settings.options.activation,
#        kernel_regularizer=kreg )(model_in)
#    model = Conv3D( \
#        filters=filters,
#        kernel_size=(1,1,1),
#        padding='same',
#        activation='linear',
#        kernel_regularizer=wreg,
#        use_bias=True)(model)
    model = Conv3D( \
        filters=filters,
        kernel_size=(3,3,3),
        padding='same',
        activation=settings.options.activation,
        kernel_regularizer=kreg,
        use_bias=True)(model_in)
    if drop:
        model = SpatialDropout3D(settings.options.dropout)(model)
    if add:
        model = Add()([model_in, model])
    return model

def module_down(model):
    model = Block(model)
    model = MaxPooling3D(pool_size=(2,2,1), strides=(2,2,1))(model)
    model = Block(model)
    return model

def module_up(model):
    model = Block(model)
    model = UpSampling3D(size=(2,2,1))(model)
    model = Block(model)
    return model

def module_mid(model, depth):
    if depth==0:
        model = Block(model)
        model = Block(model)
        return model
    else:
        m_down = module_down(model)
        m_mid  = module_mid(m_down, depth=depth-1)
        m_up   = module_up(m_mid)
        m_up   = Block(m_up, add=False)
        m_up   = Add()([model, m_up])
        m_up   = Block(m_up, add=False, drop=True)
        return m_up

def get_unet( _num_classes=1):
    _depth   = settings.options.depth
    _filters = settings.options.filters

    indim = (settings._ny, settings._nx, settings.options.thickness)
    layer_in  = Input(shape=(*indim,1))

    layer_mid = Conv3D( \
            filters=_filters,
            kernel_size=(3,3,3),
            padding='same',
            activation=settings.options.activation )(layer_in)
    layer_mid = Add()([layer_in, layer_mid])
    layer_mid = module_mid(layer_mid, depth=_depth)
    if settings.options.dropout > 0.0:
        layer_mid = SpatialDropout3D(settings.options.dropout)(layer_mid)

    layer_out = Conv3D(\
            filters=_num_classes,
            kernel_size=(1,1,1),
            padding='same',
#            activation='softmax',
            activation='sigmoid',
            use_bias=True)(layer_mid)

    model = Model(inputs=layer_in, outputs=layer_out)
    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model
