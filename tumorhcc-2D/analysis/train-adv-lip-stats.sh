#!/bin/bash

rootdir=./conv/noaug/onegpu
kf=005
idf=002
ngpu=1
options="--trainmodel --dbfile=../trainingdata.csv --rescon --numepochs=30 --gpu=$ngpu"

# delta is the adversarial perturbation
deltas='0.2 0.1 0.05 0.01 0.005'

outdir=$rootdir
ldir=$outdir/$kf/$idf/lip
mdir=$outdir/$kf/$idf/liver/modelunet.h5
mkdir -p $outdir
mkdir -p $ldir
python3 tumorhcc.py $options --kfolds=$kf --idfold=$idf --outdir=$outdir 
python3 ../analysis/kernel-analysis-2.py --model=$mdir --outdir=$ldir --gpu=$ngpu >> $ldir/lipschitz_log.txt
for delta in $deltas
do
    echo delta=$delta 
    adir=$outdir/$kf/$idf/adv/$delta
    mkdir -p $adir
    mkdir -p $adir/l2
    mkdir -p $adir/dsc
    python3 ../analysis/adversary.py --model=$mdir --outdir=$adir/ --delta=$delta
done

echo done.
