#!/bin/sh

TIME_HOME="/home/joseph/Projects/global_urban_computing/core"

TF_HOME="/home/joseph/anaconda3/envs/tensorflow/bin"

#${TF_HOME}/python -u ${TIME_HOME}/BP/ts_bp_keras.py -e 3

nohup ${TF_HOME}/python -u ${TIME_HOME}/seq2seq/ts_seq2seq_keras2.py -c > ${TIME_HOME}/seq2seq/seq2seq.log 2>&1 &

unset TIME_HOME
unset TF_HOME

