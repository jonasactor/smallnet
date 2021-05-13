#!/bin/bash

kf=005
idfolds='000 001 002 003 004'
#idfolds='000'

rootdir=./model-comparisons-ttdo
options="--gpu=2  --dbfile=/rsrch1/ip/jacctor/livermask/trainingdata.csv --trainingresample=256 --thickness=1 --numepochs=40 --depth=3 --liver --filters=16 --hu_lb=-100 --hu_ub=200 --datafiles_liver=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_liver.txt --datafiles_tumor=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_tumor.txt --datafiles_all=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations.txt"

#modeltypes='unet resnet densenet'
modeltypes='resnet'
for m in $modeltypes
do
	rm -rf $rootdir/$m
	rm -rf $rootdir/$m-pocket
	rm -rf $rootdir/$m-c2Dt
	rm -rf $rootdir/$m-pocket-c2Dt
	mkdir -p $rootdir/$m
	mkdir -p $rootdir/$m-pocket
	mkdir -p $rootdir/$m-c2Dt
	mkdir -p $rootdir/$m-pocket-c2Dt
	for idf in $idfolds
	do
		echo starting modeltype $m fold $idf
# these do not save output to file
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m  --predictmodel=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/model-comparisons/$m/005/$idf/liver/modelunet.h5 --predictfromcsv=/rsrch1/ip/jacctor/livermask/trainingdata.csv --ttdo
		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket --pocket --predictmodel=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/model-comparisons/$m-pocket/005/$idf/liver/modelunet.h5 --predictfromcsv=/rsrch1/ip/jacctor/livermask/trainingdata.csv --ttdo
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m 
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-c2Dt --conv2Dtranspose 
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket-c2Dt --pocket --conv2Dtranspose 
# these save output to txt file
#               python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m > $rootdir/$m/output-$kf-$idf.txt 
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket --pocket > $rootdir/$m-pocket/output-$kf-$idf.txt    
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-c2Dt --conv2Dtranspose > $rootdir/$m-c2Dt/output-$kf-$idf.txt
#		python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket-c2Dt --pocket --conv2Dtranspose > $rootdir/$m-pocket-c2Dt/output-$kf-$idf.txt
	done
done


echo done.
