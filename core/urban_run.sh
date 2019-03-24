#!/bin/sh

TIME_HOME="/home/joseph/Projects/time_series_forecast"

TF_HOME="/home/joseph/anaconda3/envs/tensorflow/bin"

#${TF_HOME}/python -u ${TIME_HOME}/BP/ts_bp_keras.py -e 3

nohup ${TF_HOME}/python -u ${TIME_HOME}/ARIMA/ts_arima.py > ${TIME_HOME}/ARIMA/arima.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/BP/ts_bp_keras.py > ${TIME_HOME}/BP/bp.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/LSTM/ts_lstm_keras.py -c > ${TIME_HOME}/LSTM/lstm_c.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/LSTM/ts_lstm_keras.py -g > ${TIME_HOME}/LSTM/lstm_g.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/LSTM/ts_stack_lstm_keras.py > ${TIME_HOME}/LSTM/lstm_stack.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/Seq2seq/ts_seq2seq_keras.py -c > ${TIME_HOME}/Seq2seq/seq2seq.log 2>&1 &

nohup ${TF_HOME}/python -u ${TIME_HOME}/Seq2seq/ts_seq2seq_keras.py -g > ${TIME_HOME}/Seq2seq/seq2seq.log 2>&1 &

unset TIME_HOME
unset TF_HOME

