#!/bin/bash

kf=005

rootdir=./model-comparisons
options="--gpu=2  --dbfile=/rsrch1/ip/jacctor/livermask/trainingdata.csv --trainingresample=256 --trainmodel --thickness=1 --numepochs=40 --depth=3 --liver --filters=16 --hu_lb=-100 --hu_ub=200 --datafiles_liver=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_liver.txt --datafiles_tumor=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_tumor.txt --datafiles_all=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations.txt"

rm -rf ./model-comparisons/unet-c2Dt/005/001
rm -rf ./model-comparisons/unet-c2Dt/output-005-001.txt
python3 tumorhcc.py --unet --conv2Dtranspose $options --kfolds=$kf --idfold=001 --outdir=$rootdir/unet-c2Dt > ./model-comparisons/unet-c2Dt/output-$kf-001.txt

rm -rf ./model-comparisons/unet-c2Dt/005/002
rm -rf ./model-comparisons/unet-c2Dt/output-005-002.txt
python3 tumorhcc.py --unet --conv2Dtranspose $options --kfolds=$kf --idfold=002 --outdir=$rootdir/unet-c2Dt > ./model-comparisons/unet-c2Dt/output-$kf-002.txt

rm -rf ./model-comparisons/resnet/005/000
rm -rf ./model-comparisons/resnet/output-005-000.txt
python3 tumorhcc.py --resnet $options --kfolds=$kf --idfold=000 --outdir=$rootdir/resnet > ./model-comparisons/resnet/output-$kf-000.txt

rm -rf ./model-comparisons/densenet/005/000
rm -rf ./model-comparisons/densenet/output-005-000.txt
python3 tumorhcc.py --densenet $options --kfolds=$kf --idfold=000 --outdir=$rootdir/densenet > ./model-comparisons/densenet/output-$kf-000.txt

echo done.
