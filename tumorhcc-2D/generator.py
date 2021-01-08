import numpy as np
import keras
from keras.utils import to_categorical
import os
import csv
from scipy import ndimage

import settings
import preprocess


def save_img_into_substacks(Xall, Yall, Xloc, Yloc, n_classes=1):
    """
    Saves 3D subset blocks as .npy files
    """
    assert Xall.shape[2] == Yall.shape[2]
    n_valid_imgs = Xall.shape[2] 
    Xfilelist = [None]*n_valid_imgs
    Yfilelist = [None]*n_valid_imgs
    for z in range(n_valid_imgs):
        Xthis = Xall[...,z,np.newaxis]
        if n_classes == 1:
            Ythis = Yall[...,z,np.newaxis]
        else:
            Ythis = to_categorical(Yall[...,z], num_classes=n_classes)
        Xthisloc = Xloc+'-'+str(z)+'.npy'
        Ythisloc = Yloc+'-'+str(z)+'.npy'
        np.save(Xthisloc, Xthis)
        np.save(Ythisloc, Ythis)
        Xfilelist[z] = Xthisloc
        Yfilelist[z] = Ythisloc
    return [x for x in Xfilelist if x is not None], [y for y in Yfilelist if y is not None]

def isolate_ROI(X, Y):
    """
    Returns slices of X,Y that contain non-background (label==0) features
    """
    nslice = X.shape[2]
    assert X.shape[2] == Y.shape[2]
    print('sizes :\t', X.shape, Y.shape)

    pad = 0

    maxY = np.amax(Y, axis=(0,1))
    if settings.options.liver:
        ROI  = maxY > 0
    elif settings.options.tumor:
        ROI  = maxY > 1
    else:
        ROI  = maxY > 0
    idx  = np.nonzero(ROI)
    try:
        min_ROI = max([np.min(idx)   - pad, 0])
        max_ROI = min([np.max(idx)+1 + pad, len(maxY)])
    except:
        min_ROI = len(maxY)//2    - pad
        max_ROI = len(maxY)//2 +1 + pad
    assert max_ROI - min_ROI >= settings.options.thickness 
    print('ROI   :\tslices', min_ROI, '-', max_ROI)
    return X[...,min_ROI:max_ROI], Y[...,min_ROI:max_ROI]


def setup_training_from_file():

    datacsv = settings.options.dbfile
    # create  custom data frame database type
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    logfileoutputdir = settings.options.outdir
    os.system ('mkdir -p ' + logfileoutputdir)
    os.system ('mkdir -p ' + logfileoutputdir + '/data')
    os.system ('mkdir -p ' + logfileoutputdir + '/data/img')
    os.system ('mkdir -p ' + logfileoutputdir + '/data/seg')
    print("Output to\t", logfileoutputdir)
    imgdir = logfileoutputdir+'/data/img'
    segdir = logfileoutputdir+'/data/seg'

    imglist = []
    seglist = []
    dataidlist = []
    with open(datacsv, 'r') as csvfile:
        myreader = csv.DictReader(csvfile, delimiter=',')
        for row in myreader:
            dataid = int(row['dataid'])
            imagelocation = '%s/%s' % (settings.options.rootlocation,row['image'])
            truthlocation = '%s/%s' % (settings.options.rootlocation,row['label'])
            print(imagelocation,truthlocation )

            numpyimage, orig_header, numpytruth  = preprocess.reorient(imagelocation, segloc=truthlocation)
            resimage = preprocess.resize_to_nn(numpyimage, transpose=False).astype(settings.IMG_DTYPE)
            resimage = preprocess.window(resimage, settings.options.hu_lb, settings.options.hu_ub)
            resimage = preprocess.rescale(resimage, settings.options.hu_lb, settings.options.hu_ub)

            restruth = preprocess.resize_to_nn(numpytruth, transpose=False).astype(settings.SEG_DTYPE)

            imgROI, segROI = isolate_ROI(resimage, restruth)
            Xloc = imgdir+'/volume-'+str(dataid)
            Yloc = segdir+'/segmentation-'+str(dataid)
            Xlist, Ylist = save_img_into_substacks(imgROI, segROI, Xloc, Yloc)
            imglist += Xlist
            seglist += Ylist
            dataidlist += [dataid]*len(Xlist)

    savelistsloc = logfileoutputdir+'/data/datalocations.txt'
    with open(savelistsloc, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['dataid','imgloc','segloc'])
        for row in zip(dataidlist, imglist, seglist):
            writer.writerow(row)

    return savelistsloc

class NpyDataGenerator(keras.utils.Sequence):
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
        Y = np.empty((self.batch_size, *self.dim, self.n_classes),  dtype=settings.SEG_DTYPE)

        # Generate data
        for i, locpair in enumerate(zip(list_x_temp, list_y_temp)):
            try:
                (xloc, yloc) = locpair
                X[i,...] = np.load(xloc)
                if settings.options.liver:
                    Y[i,...] = np.load(yloc)
#                    Y[i,...] = (np.load(yloc) >= 1).astype(settings.SEG_DTYPE)
                elif settings.options.tumor:
                    Y[i,...] = (np.load(yloc) > 1).astype(settings.SEG_DTYPE)
            except:
                X[i,...] = np.zeros((*self.dim, self.n_channels)) - 1.0
                Y[i,...] = np.zeros((*self.dim, self.n_classes))

        return X, Y
