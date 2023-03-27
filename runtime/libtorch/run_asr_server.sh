#!/usr/bin/env bash


export GLOG_logtostderr=1
export GLOG_v=2

#必备资源
model_dir=../resource/ASR                 #端到端ASR识别模型文件路径，模型为final.zip， 量化模型为final_quant.zip
unit_path=../resource/ASR/units.txt   #端到端ASR识别模型对应的建模单元路径

# VAD 参数设置
min_trailing_silence=800                  #VAD拖尾静音长度，单位ms，语音间隔的静音段超过此长度，则语音被截断(当前结果作为final_result返回)。
max_utterance_length=20000                #VAD最大截断长度，单位ms，连续语音长度(中间无静音段)超过此长度，则被截断(当前结果作为final_result返回)。

#可选资源：NgramLM， 提供fst_path具体路径，即可使用语言模型进行解码。
lang_dir=../resource/ASR        #Ngram语言模型路径
fst_path=$lang_dir/TLG.fst    #语言模型词图路径，默认名称都是TLG.fst，如果提供具体路径，则解码时会自动加载
dict_path=$lang_dir/words.txt  #语言模型词图对应词典的路径，默认名称为words.txt和TLG.fst放在同一个路径下

#可选资源：热词列表，提供context_path，即可加载热词，并在解码中进行热词激励。
context_path=$lang_dir/hot_words.txt    #热词路径，每个词一行
context_score=3                         #热词激励分数，一般不用改

#下面是WFST解码参数，一般不用改。
acoustic_scale=1.0
beam=15.0
lattice_beam=7.5
min_active=200
max_active=7000
blank_skip_thresh=0.98
length_penalty=-3.0
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
  echo $wfst_decode_opts > ./wfst_config.txt
fi

decode_opts=
if [ ! -z $context_path ]; then
  decode_opts="--context_path ${context_path} --context_score ${context_score} "
fi

CUDA_VISIBLE_DEVICES="0" ./build/bin/websocket_server_main \
    --port 10086 \
    --frame_shift 100 \
    --chunk_size 16 ${wfst_decode_opts} ${decode_opts} \
    --ctc_weight 0.5 --rescoring_weight 1.0 \
    --reverse_weight 0.3 \
    --min_trailing_silence $min_trailing_silence  \
    --max_utterance_length $max_utterance_length  \
    --model_path $model_dir/final.zip \
    --unit_path $unit_path 2>&1 | tee server.log