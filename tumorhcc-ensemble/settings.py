###
### process options and setup global variables
###

def process_options():

    from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)

    parser = OptionParser()
    parser.add_option( "--hvd",
                  action="store_true", dest="with_hvd", default=False,
                  help="use horovod for multicore parallelism")
    parser.add_option( "--gpu",
                  type="int", dest="gpu", default=0,
                  help="number of gpus", metavar="int")

    parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="Path")
    parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model on all data", metavar="bool")
    parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="model weights (.h5) for prediction", metavar="Path")
    parser.add_option( "--predictfromcsv",
                  action="store", dest="predictfromcsv", default=None,
                  help="csv of files to predict", metavar="csv")
    parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="image to segment", metavar="Path")
    parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="location for seg prediction output ", metavar="Path")
    parser.add_option( "--ttdo",
                  action="store_true", dest="ttdo", default=False,
                  help="perform multiple evaluations of predictions using different dropout draws", metavar="bool")
    parser.add_option( "--ntrials",
                  type="int", dest="ntrials", default=10,
                  help="number of Bernoulli trials for TTDO dropout draws", metavar="int")

    parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
    parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="root location for images for training; prepend to csv", metavar="Path")
    parser.add_option("--datafiles",
                  action="store", dest="datafiles", default=None,
                  help="location of pre-saved files", metavar="path")
    parser.add_option("--datafiles_liver",
                  action="store", dest="datafiles_liver", default=None,
                  help="location of pre-saved files for liver training", metavar="path")
    parser.add_option("--datafiles_tumor",
                  action="store", dest="datafiles_tumor", default=None,
                  help="location of pre-saved files for tumor training", metavar="path")
    parser.add_option("--datafiles_all",
                  action="store", dest="datafiles_all", default=None,
                  help="location of pre-saved files for prediction", metavar="path")

    parser.add_option("--liver",
                  action="store_true", dest="liver", default=False,
                  help="perform liver segmentation", metavar="bool")
    parser.add_option("--tumor",
                  action="store_true", dest="tumor", default=False,
                  help="perform tumor segmentation", metavar="bool")

    parser.add_option("--thickness",
                  type="int", dest="thickness", default=5,
                  help='3d thickness', metavar="int")
    parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
    parser.add_option( "--hu_lb",
                  type="int", dest="hu_lb", default=-100,
                  help="lower bound for CT windowing", metavar="int")
    parser.add_option( "--hu_ub",
                  type="int", dest="hu_ub", default=400,
                  help="upper bound for CT windowing", metavar="int")
    parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=1,
                  help="perform kfold prediction with k folds", metavar="int")
    parser.add_option( "--idfold",
                  type="int", dest="idfold", default=-1,
                  help="individual fold for k folds", metavar="int")

    parser.add_option("--unet",
                  action="store_true", dest="unet", default=False,
                  help="use UNet architecture", metavar="bool")
    parser.add_option("--resnet",
                  action="store_true", dest="resnet", default=False,
                  help="use ResNet architecture", metavar="bool")
    parser.add_option("--densenet",
                  action="store_true", dest="densenet", default=False,
                  help="use DenseNet architecture", metavar="bool")
    parser.add_option("--pocket",
                  action="store_true", dest="pocket", default=False,
                  help="use PocketNet (i.e. adding instead of concatenating)", metavar="bool")

    parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=30,
                  help="number of epochs for training", metavar="int")
    parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adam',
                  help="setup info", metavar="string")
    parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=16,
                  help="batch size", metavar="int")
    # parser.add_option( "--validationbatch",
    #               type="int", dest="validationbatch", default=16,
    #               help="batch size", metavar="int")
    parser.add_option( "--depth",
                  type="int", dest="depth", default=4,
                  help="number of down steps to UNet", metavar="int")
    parser.add_option( "--filters",
                  type="int", dest="filters", default=16,
                  help="number of filters for output of CNN layer", metavar="int")
    parser.add_option( "--activation",
                  action="store", dest="activation", default='relu',
                  help="activation function", metavar="string")
    parser.add_option( "--dropout",
                  type="float", dest="dropout", default=0.25,
                  help="percent  dropout", metavar="float")
    parser.add_option( "--l1reg",
                  type="float", dest="l1reg", default=0.0,
                  help="regularize with entrywise-l1 norm on kernels", metavar="float")
    parser.add_option( "--lr",
                  type="float", dest="lr", default=0.001,
                  help="learning rate for Adam optimizer. Not used if not using Adam.", metavar="float")
    parser.add_option( "--ista",
                  type="float", dest="ista", default=0.0,
                  help="use sparse thresholding to enforce kernel sparsity", metavar="bool")
    parser.add_option("--depthwise",
                  action="store_true", dest="depthwise", default=False,
                  help="use DepthwiseConv2D( (3,3) ) followed by Conv2D( (1,1) ) kernels", metavar="bool")
    parser.add_option("--conv2Dtranspose",
                  action="store_true", dest="conv2Dtranspose", default=False,
                  help="use 2D transpose convolutions to upsample", metavar="bool")

    parser.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="verbose printing during training", metavar="bool")
    global options
    global args

    (options, args) = parser.parse_args()

    return (options, args)


def perform_setup(options):

    import numpy as np
    import sys
    import keras
    import keras.backend as K
    import tensorflow as tf

    print("Performing setup")

    sys.setrecursionlimit(5000)

    if options.with_hvd:
        import horovod.keras as hvd
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        if options.gpu > 1:
            devlist = '0'
            for i in range(1,options.gpu):
                devlist += ','+str(i)
            config.gpu_options.visible_device_list = devlist
        else:
            config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction=0.5
    K.set_session(tf.Session(config=config))


    global _globalnpfile
    global _globalexpectedpixel
    global IMG_DTYPE
    global SEG_DTYPE
    global FLOAT_DTYPE
    global _nx
    global _ny


    # raw dicom data is usually short int (2bytes) datatype
    # labels are usually uchar (1byte)
    IMG_DTYPE = np.int16
    SEG_DTYPE = np.uint8
    FLOAT_DTYPE = np.float32

    # _globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
    # _globalexpectedpixel=512
    _nx = options.trainingresample
    _ny = options.trainingresample

    return IMG_DTYPE, SEG_DTYPE, _nx, _ny
