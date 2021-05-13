#!/bin/bash

kf=005
#idfolds='000 001 002 003 004'
idfolds='000'

rootdir=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/ttdo-adv
options="--gpu=2  --dbfile=/rsrch1/ip/jacctor/livermask/trainingdata.csv --trainingresample=256 --thickness=1 --numepochs=40 --depth=3 --liver --filters=16 --hu_lb=-100 --hu_ub=200 --datafiles_liver=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_liver.txt --datafiles_tumor=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations_tumor.txt --datafiles_all=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/alldata/data/datalocations.txt"


m='resnet'

predcsv=/rsrch1/ip/jacctor/livermask/trainingdata.csv

# delta is the adversarial perturbation
deltas='0.2 0.1 0.05 0.01 0.005'
#deltas='0.1'

rm -rf $rootdir/$m
rm -rf $rootdir/$m-pocket
mkdir -p $rootdir/$m
mkdir -p $rootdir/$m-pocket
for idf in $idfolds
do
	echo starting modeltype $m fold $idf
	mloc=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/model-comparisons/$m/005/$idf/liver/modelunet.h5
	mploc=/rsrch1/ip/jacctor/smallnet/tumorhcc-2D/model-comparisons/$m-pocket/005/$idf/liver/modelunet.h5
#	python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m  --predictmodel=$mloc --predictfromcsv=$predcsv --ttdo
#	python3 tumorhcc.py --$m $options --kfolds=$kf --idfold=$idf --outdir=$rootdir/$m-pocket --pocket --predictmodel=$mploc --predictfromcsv=$predcsv --ttdo

	for delta in $deltas
	do
		echo delta=$delta 
		adir=$rootdir/$m/$kf/$idf/adv/$delta
		mkdir -p $adir
		mkdir -p $adir/l2
		mkdir -p $adir/dsc
		python3 adversary3.py --model=$mloc --outdir=$adir/ --delta=$delta
		apdir=$rootdir/$m-pocket/$kf/$idf/adv/$delta
		mkdir -p $apdir
		mkdir -p $apdir/l2
		mkdir -p $apdir/dsc
		python3 adversary3.py --model=$mploc --outdir=$apdir/ --delta=$delta
	done
done

echo done.
