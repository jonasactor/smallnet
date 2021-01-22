#!/bin/bash

kf=005
#idfolds='000 001 002 003 004'
idfolds='000'

rootdir=./model-comparisons-ensemble-512
options="--gpu=1  --dbfile=/rsrch1/ip/jacctor/livermask/trainingdata.csv --trainingresample=512 --trainmodel --thickness=1 --numepochs=20 --depth=3 --liver --filters=16 --hu_lb=-100 --hu_ub=200 --datafiles_liver=/rsrch1/ip/jacctor/smallnet/tumorhcc-ensemble/all-data-512/data/datalocations_liver.txt --datafiles_tumor=/rsrch1/ip/jacctor/smallnet/tumorhcc-ensemble/all-data-512/data/datalocations_tumor.txt --datafiles_all=/rsrch1/ip/jacctor/smallnet/tumorhcc-ensemble/all-data-512/data/datalocations.txt --verbose"

#modeltypes='unet resnet densenet'
modeltypes='resnet'
for m in $modeltypes
do
	rm -rf $rootdir/$m
	rm -rf $rootdir/$m-pocket
#	mkdir -p $rootdir/$m
	mkdir -p $rootdir/$m-pocket
	for idf in $idfolds
	do
		echo starting modeltype $m fold $idf
# these do not save output to file
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m 
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket --pocket 
# these save output to txt file
#                python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m > $rootdir/$m/output-$kf-$idf.txt 
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket --pocket > $rootdir/$m-pocket/output-$kf-$idf.txt    
	done
done


echo done.
