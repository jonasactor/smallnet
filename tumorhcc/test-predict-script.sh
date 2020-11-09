#!/bin/bash

kf=005
idf=000

rootdir=./results/res128/batch32-filter16-depth4
options="--gpu=1  --dbfile=../trainingdata.csv --trainingresample=128"
python3 tumorhcc.py $options  --outdir=$rootdir --predictmodel=$rootdir/$kf/$idf/liver/modelunet.h5 --kfolds=$kf --idfold=$idf 

echo done.
