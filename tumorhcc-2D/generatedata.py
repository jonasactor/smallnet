import numpy as np
import os
import csv
import sys
from scipy import ndimage
import nibabel as nib
import skimage.transform

###
### SET UP OPTIONS
###

from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
def process_options():


    parser = OptionParser()

    parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="Path")
    parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
    parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="root location for images for training; prepend to csv", metavar="Path")
    parser.add_option("--liver",
                  action="store_true", dest="liver", default=False,
                  help="perform liver segmentation", metavar="bool")
    parser.add_option("--tumor",
                  action="store_true", dest="tumor", default=False,
                  help="perform tumor segmentation", metavar="bool")
    parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
    parser.add_option( "--hu_lb",
                  type="int", dest="hu_lb", default=-100,
                  help="lower bound for CT windowing", metavar="int")
    parser.add_option( "--hu_ub",
                  type="int", dest="hu_ub", default=200,
                  help="upper bound for CT windowing", metavar="int")
 
    global options
    global args
    (options, args) = parser.parse_args()
    return options, args



global IMG_DTYPE
global SEG_DTYPE
global FLOAT_DTYPE

IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8
FLOAT_DTYPE = np.float32

(options, args) = process_options()






###
### IMAGE PROCESSING FUNCTIONS
###

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
    numpyimage  = imagedata.get_data().astype(IMG_DTYPE)
    numpyseg    = None
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


# sample down to nn's expected input size
def resize_to_nn(img,transpose=False):
    if img.shape[1] == options.trainingresample and img.shape[0] == options.trainingresample:
        expected = img
    else:
        expected = skimage.transform.resize(img,
            (options.trainingresample, options.trainingresample, img.shape[2]),
            order=0,
            mode='constant',
            preserve_range=True)
    return expected









###
### GENERATE LISTS OF .NPY FILES FOR TRAINING
###

def setup_training_from_file():

    datacsv = options.dbfile

    logfileoutputdir = options.outdir
    os.system ('mkdir -p ' + logfileoutputdir)
    os.system ('mkdir -p ' + logfileoutputdir + '/data')
    os.system ('mkdir -p ' + logfileoutputdir + '/data/img')
    os.system ('mkdir -p ' + logfileoutputdir + '/data/seg')
    print("Output to\t", logfileoutputdir)
    imgdir = logfileoutputdir+'/data/img'
    segdir = logfileoutputdir+'/data/seg'

    imglist = []
    seglist = []
    imglist_liver = []
    seglist_liver = []
    imglist_tumor = []
    seglist_tumor = []
    dataidlist = []
    dataidlist_liver = []
    dataidlist_tumor = []
    with open(datacsv, 'r') as csvfile:
        myreader = csv.DictReader(csvfile, delimiter=',')
        for row in myreader:
            dataid = int(row['dataid'])
            imagelocation = '%s/%s' % (options.rootlocation,row['image'])
            truthlocation = '%s/%s' % (options.rootlocation,row['label'])
            print(imagelocation,truthlocation )

            numpyimage, orig_header, numpytruth  = reorient(imagelocation, segloc=truthlocation)
            resimage = resize_to_nn(numpyimage).astype(IMG_DTYPE)
            resimage = window(resimage, options.hu_lb, options.hu_ub)
            resimage = rescale(resimage, options.hu_lb, options.hu_ub)

            restruth = resize_to_nn(numpytruth).astype(SEG_DTYPE)


            Xloc = imgdir+'/volume-'+str(dataid).zfill(3)
            Yloc = segdir+'/segmentation-'+str(dataid).zfill(3)

            assert resimage.shape[2] == restruth.shape[2]
            n_valid_imgs = resimage.shape[2]
            Xfilelist = [None]*n_valid_imgs
            Yfilelist = [None]*n_valid_imgs
            for z in range(n_valid_imgs):
                Xthis = resimage[...,z,np.newaxis]
                Ythis = restruth[...,z,np.newaxis]
                
                Xthisloc = Xloc+'-'+str(z).zfill(4)+'.npy'
                Ythisloc = Yloc+'-'+str(z).zfill(4)+'.npy'

                np.save(Xthisloc, Xthis)
                np.save(Ythisloc, Ythis)
        
                Xfilelist[z] = Xthisloc
                Yfilelist[z] = Ythisloc

            maxval = np.amax(restruth, axis=(0,1)).tolist()
            assert len(maxval) == len(Yfilelist)

            Xliverlist = [ Xfilelist[idx] for idx in range(n_valid_imgs) if maxval[idx] > 0]
            Yliverlist = [ Yfilelist[idx] for idx in range(n_valid_imgs) if maxval[idx] > 0]
            Xtumorlist = [ Xfilelist[idx] for idx in range(n_valid_imgs) if maxval[idx] > 1]
            Ytumorlist = [ Yfilelist[idx] for idx in range(n_valid_imgs) if maxval[idx] > 1]


            imglist += Xfilelist
            seglist += Yfilelist
            imglist_liver += Xliverlist
            seglist_liver += Yliverlist
            imglist_tumor += Xtumorlist
            seglist_tumor += Ytumorlist
            dataidlist       += [dataid]*len(Xfilelist)
            dataidlist_liver += [dataid]*len(Xliverlist)
            dataidlist_tumor += [dataid]*len(Xtumorlist)

    savelistsloc = logfileoutputdir+'/data/datalocations.txt'
    print('saving   all files to', savelistsloc)
    with open(savelistsloc, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['dataid','imgloc','segloc'])
        for row in zip(dataidlist, imglist, seglist):
            writer.writerow(row)

    savelistsloc_liver = logfileoutputdir+'/data/datalocations_liver.txt'
    print('saving liver files to', savelistsloc_liver)
    with open(savelistsloc_liver, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['dataid','imgloc','segloc'])
        for row in zip(dataidlist_liver, imglist_liver, seglist_liver):
            writer.writerow(row)
    
    savelistsloc_tumor = logfileoutputdir+'/data/datalocations_tumor.txt'
    print('saving tumor files to', savelistsloc_tumor)
    with open(savelistsloc_tumor, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['dataid','imgloc','segloc'])
        for row in zip(dataidlist_tumor, imglist_tumor, seglist_tumor):
            writer.writerow(row)

    return savelistsloc, savelistsloc_liver, savelistsloc_tumor


saveloclist, saveloclist_liver, saveloclist_tumor = setup_training_from_file()
