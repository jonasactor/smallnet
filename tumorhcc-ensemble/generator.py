import numpy as np
import keras
from keras.utils import to_categorical
import os
import csv
from scipy import ndimage

import settings
import preprocess


class NpyDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgloclist,
                       segloclist,
                       batch_size=settings.options.trainingbatch,
                       dim=(settings.options.trainingresample, settings.options.trainingresample),
                       n_channels=1,
                       n_classes=3,
                       loc_csv=None,
                       shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.imgloclist = imgloclist
        self.segloclist = segloclist
        self.ndata = len(imgloclist)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.loc_csv = loc_csv
        self.on_epoch_end()
        self.indexes = np.arange(self.ndata)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.imgloclist[k] for k in indexes]
        list_y_temp = [self.segloclist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_x_temp, list_y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ndata)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=settings.FLOAT_DTYPE)
        Y = np.empty((self.batch_size, *self.dim, self.n_classes),  dtype=settings.SEG_DTYPE)

        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                X[i,...] = np.load(xloc)

                Ydata = np.load(yloc)
                Y[i,...] = to_categorical(Ydata, num_classes=3)

                # data augmentation - flip up/down
                # attempting to fix issue with k-fold setup
                coinflip = np.random.binomial(1,0.5)
                if coinflip:
                    X[i,...] = np.flipud(X[i,...])
                    Y[i,...] = np.flipud(Y[i,...])

            except:
                X[i,...] = np.zeros((*self.dim, self.n_channels)) - 1.0
                Y[i,...] = np.zeros((*self.dim, self.n_classes))

        return X, Y




class NpyDataPredictionGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgloclist,
                       segloclist,
                       batch_size=settings.options.trainingbatch,
                       dim=(settings.options.trainingresample, settings.options.trainingresample),
                       n_channels=1,
                       n_classes=3,
                       loc_csv=None,
                       shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.imgloclist = imgloclist
        self.segloclist = segloclist
        self.ndata = len(imgloclist)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.loc_csv = loc_csv
        self.on_epoch_end()
        self.indexes = np.arange(self.ndata)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size > self.ndata:
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.imgloclist[k] for k in indexes]
        list_y_temp = [self.segloclist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_x_temp, list_y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ndata)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        this_batch_size = min([self.batch_size, len(list_x_temp)])
        # Initialization
        X = np.empty((this_batch_size, *self.dim, self.n_channels), dtype=settings.FLOAT_DTYPE)
        Y = np.empty((this_batch_size, *self.dim, self.n_classes),  dtype=settings.SEG_DTYPE)

        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                X[i,...] = np.load(xloc)
                Ydata    = np.load(yloc)
                Y[i,...] = to_categorical(Ydata, num_classes=3)
            except:
                X[i,...] = np.zeros((*self.dim, self.n_channels)) - 1.0
                Y[i,...] = np.zeros((*self.dim, self.n_classes))

        return X, Y






class NpyDataGenerator_Liver(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgloclist,
                       segloclist,
                       batch_size=settings.options.trainingbatch,
                       dim=(settings.options.trainingresample, settings.options.trainingresample),
                       n_channels=1,
                       n_classes=1,
                       loc_csv=None,
                       shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.imgloclist = imgloclist
        self.segloclist = segloclist
        self.ndata = len(imgloclist)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.loc_csv = loc_csv
        self.on_epoch_end()
        self.indexes = np.arange(self.ndata)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.imgloclist[k] for k in indexes]
        list_y_temp = [self.segloclist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_x_temp, list_y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ndata)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=settings.FLOAT_DTYPE)
        Y = np.empty((self.batch_size, *self.dim, self.n_classes),  dtype=settings.FLOAT_DTYPE)

        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                X[i,...] = np.load(xloc)
                Y[i,...] = (np.load(yloc) >= 1.).astype(settings.FLOAT_DTYPE)

                # data augmentation - flip up/down
                # attempting to fix issue with k-fold setup
                coinflip = np.random.binomial(1,0.5)
                if coinflip:
                    X[i,...] = np.flipud(X[i,...])
                    Y[i,...] = np.flipud(Y[i,...])

            except:
                X[i,...] = np.zeros((*self.dim, self.n_channels)) - 1.0
                Y[i,...] = np.zeros((*self.dim, self.n_classes))

        return X, Y


class NpyDataGenerator_Tumor(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgloclist,
                       segloclist,
                       batch_size=settings.options.trainingbatch,
                       dim=(settings.options.trainingresample, settings.options.trainingresample),
                       n_channels=1,
                       n_classes=1,
                       loc_csv=None,
                       shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.imgloclist = imgloclist
        self.segloclist = segloclist
        self.ndata = len(imgloclist)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.loc_csv = loc_csv
        self.on_epoch_end()
        self.indexes = np.arange(self.ndata)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.imgloclist[k] for k in indexes]
        list_y_temp = [self.segloclist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_x_temp, list_y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ndata)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X    = np.empty((self.batch_size, *self.dim, 2*self.n_channels), dtype=settings.FLOAT_DTYPE)
        Y    = np.empty((self.batch_size, *self.dim,   self.n_classes),  dtype=settings.FLOAT_DTYPE)


        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                IMG = np.load(xloc)

                segdata = np.load(yloc)
                MASK = (segdata >= 1.).astype(settings.FLOAT_DTYPE)
                X[i,...,0] = IMG[...,0]
                X[i,...,1] = MASK[...,0]
                Y[i,...]   = (segdata >  1.).astype(settings.FLOAT_DTYPE)

                # data augmentation - flip up/down
                # attempting to fix issue with k-fold setup
                coinflip = np.random.binomial(1,0.5)
                if coinflip:
                    X[i,...,0]  = np.flipud(X[i,...,0])
                    X[i,...,1]  = np.flipud(X[i,...,1])
                    Y[i,...]    = np.flipud(Y[i,...])

            except:
                X[i,...,0]  = np.zeros(self.dim) - 1.0
                X[i,...,1]  = np.zeros(self.dim)
                Y[i,...]    = np.zeros((*self.dim, self.n_classes))

        return X,Y




class NpyDataGenerator_Ensemble(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgloclist,
                       segloclist,
                       batch_size=settings.options.trainingbatch,
                       dim=(settings.options.trainingresample, settings.options.trainingresample),
                       n_channels=1,
                       n_classes=2,
                       loc_csv=None,
                       shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.imgloclist = imgloclist
        self.segloclist = segloclist
        self.ndata = len(imgloclist)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.loc_csv = loc_csv
        self.on_epoch_end()
        self.indexes = np.arange(self.ndata)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.imgloclist[k] for k in indexes]
        list_y_temp = [self.segloclist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_x_temp, list_y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ndata)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        IMG  = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=settings.FLOAT_DTYPE)
        SEG  = np.empty((self.batch_size, *self.dim, self.n_classes),  dtype=settings.FLOAT_DTYPE)

        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                IMG[i,...]   = np.load(xloc)

                segdata = np.load(yloc)[...,0]
                SEG[i,...,0] = (segdata >= 1.).astype(settings.FLOAT_DTYPE)
                SEG[i,...,1] = (segdata >  1.).astype(settings.FLOAT_DTYPE)

                # data augmentation - flip up/down
                coinflip = np.random.binomial(1,0.5)
                if coinflip:
                    IMG[i,...]    = np.flipud(IMG[i,...])
                    SEG[i,...,0]  = np.flipud(SEG[i,...,0])
                    SEG[i,...,1]  = np.flipud(SEG[i,...,1])

            except:
                IMG[i,...] = np.zeros((*self.dim, self.n_channels)) - 1.0
                SEG[i,...] = np.zeros((*self.dim, self.n_classes))

        return IMG, SEG




class NpyDataGenerator_Prediction(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, imgloclist,
                       segloclist,
                       batch_size=settings.options.trainingbatch,
                       dim=(settings.options.trainingresample, settings.options.trainingresample),
                       n_channels=1,
                       n_classes=2,
                       loc_csv=None,
                       shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.imgloclist = imgloclist
        self.segloclist = segloclist
        self.ndata = len(imgloclist)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.loc_csv = loc_csv
        self.on_epoch_end()
        self.indexes = np.arange(self.ndata)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size > self.ndata:
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.imgloclist[k] for k in indexes]
        list_y_temp = [self.segloclist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_x_temp, list_y_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.ndata)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        this_batch_size = min([self.batch_size, len(list_x_temp)])
        # Initialization
        X = np.empty((this_batch_size, *self.dim, self.n_channels), dtype=settings.FLOAT_DTYPE)
        Y = np.empty((this_batch_size, *self.dim, self.n_classes),  dtype=settings.FLOAT_DTYPE)

        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                X[i,...] = np.load(xloc)
                Ydata    = np.load(yloc)[...,0]
                Y[i,...,0] = (Ydata >= 1.0).astype(settings.FLOAT_DTYPE)
                Y[i,...,1] = (Ydata >  1.0).astype(settings.FLOAT_DTYPE)
            except:
                X[i,...] = np.zeros((*self.dim, self.n_channels)) - 1.0
                Y[i,...] = np.zeros((*self.dim, self.n_classes))

        return X, Y

