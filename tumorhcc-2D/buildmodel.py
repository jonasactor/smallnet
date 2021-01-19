import numpy as np
import keras
from keras.layers import Conv3D, Add, SpatialDropout3D, AveragePooling3D, UpSampling3D, Input, MaxPooling3D, Concatenate
from keras.layers import Conv2D, SpatialDropout2D, AveragePooling2D, UpSampling2D, MaxPooling2D, DepthwiseConv2D, Conv2DTranspose
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

def module_down(model, filters):
    return Block(model, filters)

def module_up(model, filters):
    return Block(model, filters) 

def module_mid(model, depth, filters=settings.options.filters):
    if depth==0:
        return Block(model, filters)

    else:

        m_down = module_down(model, filters=filters)
        
        if not settings.options.pocket:
            filters*=2
            m_down = Conv2D( \
                filters=filters,
                kernel_size=(1,1),
                padding='same',
                activation=None)(m_down)
        m_mid = MaxPooling2D()(m_down)
        
        m_mid = module_mid(m_mid, depth=depth-1, filters=filters)
    
#        m_mid = module_up(m_mid, filters=filters) 

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

        m_up = module_up(m_up, filters=filters)
        return m_up




def get_unet( _num_classes=1):
    _depth   = settings.options.depth
    _filters = settings.options.filters
    if _num_classes == 1:
        act_f = 'sigmoid'
    else:
        act_f = 'softmax'

    indim = (settings._ny, settings._nx)
    layer_in  = Input(shape=(*indim,1))

    layer_features = Conv2D( \
            filters=_filters,
            kernel_size=(7,7),
            padding='same',
            activation=settings.options.activation )(layer_in)

    layer_mid = module_mid(layer_features, depth=_depth)

    layer_mid = Concatenate()([layer_features, layer_mid])
    layer_mid = SpatialDropout2D(settings.options.dropout)(layer_mid)
    layer_out = Conv2D(\
            filters=_num_classes,
            kernel_size=(1,1),
            padding='same',
            activation= act_f,
            use_bias=True)(layer_mid)

    model = Model(inputs=layer_in, outputs=layer_out)
    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model
