#!/bin/bash

. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="4,5,6,7"
stage=4 # start from 0 if you need to start from data preparation
stop_stage=4

# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
# The num of nodes or machines used for multi-machine training
# Default 1 for single machine/node
# NFS will be needed if you want run multi-machine training
num_nodes=1
# The rank of each node or machine, range from 0 to num_nodes -1
# The first node/machine sets node_rank 0, the second one sets node_rank 1
# the third one set node_rank 2, and so on. Default 0
node_rank=0

feat_dir=raw_wav

data_type=shard
num_utts_per_shard=1000
prefetch=100
cmvn_sampling_divisor=100  # 20 means 5% of the training data to estimate cmvn
train_set=train
dev_set=dev

# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
train_config=conf/train_u2++_conformer_wavaug1.yaml
# English modeling unit
# Optional 1. bpe 2. char
en_modeling_unit=bpe
dict=data/dict_$en_modeling_unit/lang_char.txt
cmvn=false   # do not use cmvn
debug=false
num_workers=2
dir=exp/conformer_wavaug1
checkpoint=

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
#average_checkpoint=false
#decode_checkpoint=$dir/10.pt
decode_nj=32
average_num=10
#decode_modes="ctc_greedy_search ctc_prefix_beam_search
#              attention attention_rescoring"
decode_modes="attention_rescoring "
context_path="data/hot_words.txt"
if [ ! -z $context_path ]; then
  decode_suffix="with_context"
else
  decode_suffix=
fi

. tools/parse_options.sh || exit 1;


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # For wav feature, just copy the data. Fbank extraction is done in training
  mkdir -p ${feat_dir}_${en_modeling_unit}
  for x in ${train_set} ${dev_set}; do
    cp -r data/$x ${feat_dir}_${en_modeling_unit}
  done

  cp -r data/test_sjtcs ${feat_dir}_${en_modeling_unit}/test  # only use sjtcs as a test set
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Compute cmvn"
  # Here we use all the training data, you can sample some some data to save time
  # BUG!!! We should use the segmented data for CMVN
  if $cmvn; then
    full_size=`cat data/${train_set}/wav.scp | wc -l`
    sampling_size=$((full_size / cmvn_sampling_divisor))
    shuf -n $sampling_size data/$train_set/wav.scp \
      > data/$train_set/wav.scp.sampled
    python3 tools/compute_cmvn_stats.py \
    --num_workers 16 \
    --train_config $train_config \
    --in_scp data/$train_set/wav.scp.sampled \
    --out_cmvn data/$train_set/global_cmvn \
    || exit 1;
  fi
fi

# This bpe model is trained on EspnetASR xmov_cs/asr2 training data set.
bpecode=conf/zh6300char_en5700bpe.model
trans_type_ops=
bpe_ops=
if [ $en_modeling_unit = "bpe" ]; then
  trans_type_ops="--trans_type cn_char_en_bpe"
  bpe_ops="--bpecode ${bpecode}"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Prepare wenet required data
  echo "Prepare data, prepare required format"
  for x in ${dev_set} ${train_set} ; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 32 ${feat_dir}_${en_modeling_unit}/$x/wav.scp \
        ${feat_dir}_${en_modeling_unit}/$x/text \
        $(realpath ${feat_dir}_${en_modeling_unit}/$x/shards) \
        ${feat_dir}_${en_modeling_unit}/$x/data.list
    else
      tools/make_raw_list.py ${feat_dir}_${en_modeling_unit}/$x/wav.scp \
      ${feat_dir}_${en_modeling_unit}/$x/text \
      ${feat_dir}_${en_modeling_unit}/$x/data.list
    fi
  done
  for x in test ; do
    tools/make_raw_list.py ${feat_dir}_${en_modeling_unit}/$x/wav.scp \
    ${feat_dir}_${en_modeling_unit}/$x/text \
    ${feat_dir}_${en_modeling_unit}/$x/data.list
  done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Training
  mkdir -p $dir
  checkpoint=$dir/1.pt
  INIT_FILE=$dir/ddp_init
  # You had better rm it manually before you start run.sh on first node.
  # rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  # The total number of processes/gpus, so that the master knows
  # how many workers to wait for.
  # More details about ddp can be found in
  # https://pytorch.org/tutorials/intermediate/dist_tuto.html
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp ${feat_dir}_${en_modeling_unit}/$train_set/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`

    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data ${feat_dir}_${en_modeling_unit}/$train_set/data.list \
      --cv_data ${feat_dir}_${en_modeling_unit}/$dev_set/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 2 \
      $cmvn_opts \
      --pin_memory \
      --bpe_model ${bpecode}
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=16
  ctc_weight=0.5
  idx=0
  reverse_weight=0.0

  nj=$decode_nj
  for mode in ${decode_modes}; do
  {
    test_dir="$dir/"`
      `"test_${mode}${decoding_chunk_size:+_chunk$decoding_chunk_size}/test"
    mkdir -p $test_dir
    mkdir -p $test_dir/split${nj}

    # Step 1. Split wav.scp
    split_scps=""
    for n in $(seq ${nj}); do
      split_scps="${split_scps} ${test_dir}/split${nj}/data.${n}.list"
    done
    scp=${feat_dir}_${en_modeling_unit}/test/data.list
    tools/data/split_scp.pl ${scp} ${split_scps}

    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
    # Step 2. Parallel decoding
    for n in $(seq ${nj}); do
    {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$idx+1])
    python wenet/bin/recognize.py --gpu $gpu_id \
      --mode $mode \
      --config $dir/train.yaml \
      --data_type "raw" \
      --test_data  "${test_dir}/split${nj}/data.${n}.list" \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size} \
      --bpe_model ${bpecode} \
      --result_file $test_dir/split${nj}/${n}.text_${en_modeling_unit} \
      &> ${test_dir}/split${nj}/${n}.log
    } &
    ((idx+=1))
    if [[ $idx -ge $num_gpus ]]; then
      idx=0
    fi
    done
    wait

    # Step 3. Merge files
    for n in $(seq ${nj}); do
      cat ${test_dir}/split${nj}/${n}.text_${en_modeling_unit}
    done > ${test_dir}/text_${en_modeling_unit}

    if [ $en_modeling_unit == "bpe" ]; then
      tools/spm_decode --model=${bpecode} --input_format=piece \
      < $test_dir/text_${en_modeling_unit} | sed -e "s/▁/ /g" > $test_dir/text
    else
      cat $test_dir/text_${en_modeling_unit} \
      | sed -e "s/▁/ /g" > $test_dir/text
    fi
    # Cer used to be consistent with kaldi & espnet
    python tools/compute-cer.py --char=1 --v=1 \
      ${feat_dir}_${en_modeling_unit}/test/text $test_dir/text > $test_dir/wer
  } &
  done
  wait
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/10.pt \
    --output_file $dir/final.zip \
    --output_quant_file $dir/final_quant.zip
fi

# Optionally, you can add LM and test it with runtime.
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  if [ ! -f data/lang_test/TLG.fst ]; then
  # 7.1 Prepare dict
  unit_file=data/dict_bpe/tokens.txt
  mkdir -p data/local/dict
  cp $unit_file data/local/dict/units.txt
  tools/fst/prepare_dict.py $unit_file data/lexicon/lexicon.txt \
    data/local/dict/lexicon.txt ${bpecode}
  # 7.2 Train lm
  lm=data/local/lm
  mkdir -p $lm
  cp data/train/text0  $lm/text
  local/aishell_train_lms.sh
  # 7.3 Build decoding TLG
  tools/fst/compile_lexicon_token_fst.sh \
    data/local/dict data/local/tmp data/local/lang
  tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
  fi
  # 7.4 Decoding with runtime
  use_lm=1
  if [ $use_lm -eq 1 ]; then
  echo "decode with TLG.fst.."
  chunk_size=16
  ./tools/decode.sh --nj 16  --frame_shift 100 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --fst_path data/lang_test/TLG.fst \
    --dict_path data/lang_test/words.txt \
    --context_path $context_path \
    --context_score 3 \
    data/test/wav.scp data/test/text $dir/final.zip \
    data/lang_test/units.txt $dir/lm_with_runtime_${chunk_size}_${decode_suffix}
  # Please see $dir/lm_with_runtime for wer
  elif [ $use_lm -eq 0 ]; then
  echo "decode without TLG.fst.."
  chunk_size=16
  ./tools/decode.sh --nj 16 --frame_shift 100 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5  --rescoring_weight 1.0 \
    --chunk_size $chunk_size \
    --context_path $context_path \
    --context_score 3 \
    data/test/wav.scp data/test/text $dir/final.zip \
    $dict $dir/runtime_${chunk_size}_${decode_suffix}
  # Please see $dir/runtime for wer
  fi
fi
