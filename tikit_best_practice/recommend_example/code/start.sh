#!/bin/bash
set -x
export CUDA_VISIBLE_DEVICES=""
export PYTHON=python3

rec_model="/opt/ml/input/data/cfs"

python3 "$@" --train_file=/opt/ml/input/data/criteo/train_new.txt \
--eval_file=/opt/ml/input/data/criteo/val.txt --model_dir=$rec_model

sleep 10
ls $rec_model
mv $rec_model/export /opt/ml/model
rm -rf $rec_model/*

exit 0