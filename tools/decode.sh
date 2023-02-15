#!/usr/bin/env bash
# Copyright 2021 Mobvoi Inc. All Rights Reserved.
# Author: binbinzhang@mobvoi.com (Binbin Zhang)
export GLOG_logtostderr=1
export GLOG_v=2

set -e

gpu_devices=
thread_per_device=1
warmup=0
nj=1
frame_shift=160   #add different frame_shift support
frame_length=400
chunk_size=-1
ctc_weight=0.0
reverse_weight=0.0
rescoring_weight=1.0
# For CTC WFST based decoding
fst_path=
dict_path=
acoustic_scale=1.0
beam=15.0
lattice_beam=12.0
min_active=200
max_active=7000
blank_skip_thresh=1.0
length_penalty=0.0

#context_biasing related
context_path=
context_score=3


. tools/parse_options.sh || exit 1;
if [ $# != 5 ]; then
  echo "Usage: $0 [options] <wav.scp> <label_file> <model_file> <unit_file> <output_dir>"
  exit 1;
fi

if ! which decoder_main > /dev/null; then
  echo "decoder_main is not built, please go to runtime/libtorch to build it."
  exit 1;
else
  decoder_tool=`which decoder_main`
  echo decoder path is $decoder_tool
fi

scp=$1
label_file=$2
model_file=$3
unit_file=$4
dir=$5

decode_opts=
if [ ! -z $context_path ]; then
  decode_opts="--context_path ${context_path} --context_score ${context_score} "
fi

mkdir -p $dir/split${nj}

# Step 1. Split wav.scp
split_scps=""
for n in $(seq ${nj}); do
  split_scps="${split_scps} ${dir}/split${nj}/wav.${n}.scp"
done
tools/data/split_scp.pl ${scp} ${split_scps}

# Step 2. Parallel decoding
wfst_decode_opts=
if [ ! -z $fst_path ]; then
  wfst_decode_opts="--fst_path $fst_path"
  wfst_decode_opts="$wfst_decode_opts --beam $beam"
  wfst_decode_opts="$wfst_decode_opts --dict_path $dict_path"
  wfst_decode_opts="$wfst_decode_opts --lattice_beam $lattice_beam"
  wfst_decode_opts="$wfst_decode_opts --max_active $max_active"
  wfst_decode_opts="$wfst_decode_opts --min_active $min_active"
  wfst_decode_opts="$wfst_decode_opts --acoustic_scale $acoustic_scale"
  wfst_decode_opts="$wfst_decode_opts --blank_skip_thresh $blank_skip_thresh"
  wfst_decode_opts="$wfst_decode_opts --length_penalty $length_penalty"
  echo $wfst_decode_opts > $dir/config
fi

idx=0
num_gpus=$(echo $gpu_devices | awk -F "," '{print NF}')

for n in $(seq ${nj}); do
{

  gpu_id=$(echo $gpu_devices | cut -d',' -f$[$idx+1])
  CUDA_VISIBLE_DEVICES=$gpu_id \
  decoder_main --thread_num $thread_per_device \
     --warmup $warmup   \
     --frame_shift $frame_shift \
     --rescoring_weight $rescoring_weight \
     --ctc_weight $ctc_weight \
     --reverse_weight $reverse_weight \
     --chunk_size $chunk_size \
     --wav_scp ${dir}/split${nj}/wav.${n}.scp \
     --model_path $model_file \
     --unit_path $unit_file \
     $decode_opts  $wfst_decode_opts \
     --result ${dir}/split${nj}/${n}.text &> ${dir}/split${nj}/${n}.log
} &
    ((idx+=1))
    if [[ $idx -ge $num_gpus ]]; then
      idx=0
    fi
done
wait

# Step 3. Merge files
for n in $(seq ${nj}); do
  cat ${dir}/split${nj}/${n}.text
done > ${dir}/text
tail $dir/split${nj}/*.log | grep RTF | awk '{sum+=$NF}END{print sum/NR}' > $dir/rtf

# Step 4. Compute WER
if [ -f $label_file ];then
python3 tools/compute-wer.py --char=1 --v=1 \
  $label_file $dir/text > $dir/wer
fi