#!/bin/bash

kf=005
idf=004
rootdir=./results/res128/batch32-filter16-depth3
options="--gpu=1  --trainmodel --dbfile=../trainingdata.csv --rescon  --numepochs=30 --trainingbatch=32 --validationbatch=32 --filters=16 --trainingresample=128 --depth=3"
python3 tumorhcc.py $options --kfolds=$kf --idfold=$idf --outdir=$rootdir

echo done.
