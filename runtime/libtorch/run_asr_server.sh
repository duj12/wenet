#!/usr/bin/env bash


export GLOG_logtostderr=1
export GLOG_v=2

model_dir=../../examples/aishell/s0/exp/conformer
lang_dir=../../examples/aishell/s0/data/lang_test

fst_path=$lang_dir/TLG.fst
dict_path=$lang_dir/words.txt

acoustic_scale=1.0
beam=15.0
lattice_beam=7.5
min_active=200
max_active=7000
blank_skip_thresh=0.98
length_penalty=0.0
wfst_decode_opts=
if [ ! -z fst_path ]; then
  wfst_decode_opts="--fst_path $fst_path"
  wfst_decode_opts="$wfst_decode_opts --beam $beam"
  wfst_decode_opts="$wfst_decode_opts --dict_path $dict_path"
  wfst_decode_opts="$wfst_decode_opts --lattice_beam $lattice_beam"
  wfst_decode_opts="$wfst_decode_opts --max_active $max_active"
  wfst_decode_opts="$wfst_decode_opts --min_active $min_active"
  wfst_decode_opts="$wfst_decode_opts --acoustic_scale $acoustic_scale"
  wfst_decode_opts="$wfst_decode_opts --blank_skip_thresh $blank_skip_thresh"
  wfst_decode_opts="$wfst_decode_opts --length_penalty $length_penalty"
  echo $wfst_decode_opts > ./wfst_config.txt
fi

./build/bin/websocket_server_main \
    --port 10086 \
    --chunk_size 16 $wfst_decode_opts \
    --ctc_weight 0.5 --rescoring_weight 1.0 \
    --model_path $model_dir/final.zip \
    --unit_path $lang_dir/units.txt 2>&1 | tee server.log