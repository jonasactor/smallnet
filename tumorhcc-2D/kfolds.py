import settings
from trainmodel import TrainModel
from predictmodel import PredictKFold

################################
# Perform K-fold validation
################################
def OneKfold(i=0, saveloclist=None):
    modelloc = TrainModel(idfold=i, saveloclist=saveloclist)
    PredictKFold(modelloc, settings.options.dbfile, settings.options.outdir, kfolds=settings.options.kfolds, idfold=i, saveloclist=saveloclist)

def Kfold(saveloclist=None):
    for iii in range(settings.options.kfolds):
        OneKfold(i=iii, saveloclist=saveloclist)
