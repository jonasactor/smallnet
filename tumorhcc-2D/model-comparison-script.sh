#!/bin/bash

kf=005
idfolds='000 001 002 003 004'


rootdir=./model-comparisons
options="--gpu=2  --dbfile=/rsrch1/ip/jacctor/livermask/trainingdata.csv --trainingresample=256 --datafiles=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/model-comparisons/test/data/datalocations.txt --trainmodel --thickness=1 --numepochs=20 --depth=4 --liver --filters=16"

modeltypes= 'unet resnet densenet'
for m in $modeltypes
do
	rm -rf $rootdir/$m
	rm -rf $rootdir/$m-pocket
	rm -rf $rootdir/$m-c2Dt
	rm -rf $rootdir/$m-pocket-c2Dt
	for idf in $idfolds
	do	    
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m      
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket      --pocket     
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-c2Dt        --conv2Dtranspose
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$Idf --outdir=$rootdir/$m-pocket-c2Dt --pocket --conv2Dtranspose
	done
done


echo done.
