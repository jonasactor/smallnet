#!/bin/bash

kf=005
#idfolds='000 001 002 003 004'
idfolds='000'

rootdir=./model-comparisons-depthwise
options="--gpu=2  --dbfile=/rsrch1/ip/jacctor/livermask/trainingdata.csv --trainingresample=256 --trainmodel --thickness=1 --numepochs=30 --depth=3 --liver --filters=16 --hu_lb=-100 --hu_ub=200 --datafiles_liver=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_liver.txt --datafiles_tumor=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_tumor.txt --datafiles_all=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations.txt --depthwise"

#modeltypes='unet resnet densenet'
modeltypes='resnet'
for m in $modeltypes
do
	mkdir -p $rootdir/$m-dm1
	mkdir -p $rootdir/$m-pocket-dm1
        mkdir -p $rootdir/$m-dm4
        mkdir -p $rootdir/$m-pocket-dm4
	for idf in $idfolds
	do
		echo starting modeltype $m fold $idf
# these do not save output to file
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-dm1 --dm=1 --verbose 
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket-dm1 --pocket --dm=1
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-dm4 --dm=4
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket-dm4 --pocket --dm=4
# these save output to txt file
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-dm1 --dm=1 > $rootdir/$m-dm1/output-$kf-$idf.txt
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket-dm1 --pocket --dm=1 > $rootdir/$m-pocket-dm1/output-$kf-$idf.txt
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-dm4 --dm=4 > $rootdir/$m-dm4/output-$kf-$idf.txt
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket-dm4 --pocket --dm=4 > $rootdir/$m-pocket-dm4/output-$kf-$idf.txt
	done
done


echo done.
