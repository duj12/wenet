#!/usr/bin/env bash


export GLOG_logtostderr=1
export GLOG_v=2

wav_path=./BAC009S0764W0402.wav

./build/bin/websocket_client_main \
    --hostname 192.168.89.52 --port 10086 \
    --wav_path $wav_path 2>&1 | tee client.log