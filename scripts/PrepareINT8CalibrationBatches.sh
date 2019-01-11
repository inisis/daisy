#!/bin/bash
if [ -z "$1" ]
then
    TEMP_DIR=./
else
    TEMP_DIR=$1/
fi

PASCAL_VOC2007_DATASET=$TEMP_DIR/VOCtrainval_06-Nov-2007.tar
tar -xf $PASCAL_VOC2007_DATASET -C $TEMP_DIR

python batchPrepare.py --inDir $TEMP_DIR/VOCdevkit/VOC2007/JPEGImages/ --outDir $TEMP_DIR