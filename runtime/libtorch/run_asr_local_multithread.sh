#!/usr/bin/env bash

thread_num=$1    # 线程数量作为脚本第一个入参传入

export GLOG_logtostderr=1
export GLOG_v=2

#必备资源，CPU部署建议使用final_quant.zip, 识别率性能只有约1%的相对下降，但是解码并发数量能够提升超过1倍（CPU20路->40路以上）。
# 注：当前pytorch官方量化模型只支持CPU推理，因此如果GPU推理, 则模型文件需选择非量化模型final.zip
model_path=../resource/ASR/final.zip    #端到端ASR识别模型文件路径，模型为final.zip， 量化模型为final_quant.zip
unit_path=../resource/ASR/units.txt   #端到端ASR识别模型对应的建模单元路径

#设定测试集
test_set=test_xmov_youling
wav_path=../resource/WAV/$test_set   #测试数据wav所在路径：需保证音频为16k采样率，单通道，16bit的wav文件
wav_scp=../resource/WAV/$test_set.scp  #此文件格式为 “音频名称  音频所在路径”， 需确保音频路径能够被当前脚本所访问
if [[ ! -f $wav_scp ]] ; then  #生成输入所需的scp格式文件
  find $wav_path -name "*.wav" | awk -F"/" -v name="" '{name=$NF; gsub(".wav", "", name); print name" "$0 }' | sort > $wav_scp
fi
decode_result=../resource/WAV/$test_set.asr   #解码结果保存路径
label_file=../resource/WAV/$test_set.txt      #如果当前音频有对应的标注文本，则提供此路径，否则直接注释掉即可
wer_result=../resource/WAV/$test_set.wer      #当提供标注文本时，会对解码结果计算准确率，保存到这个路径下

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


warmup=1   #这个参数是在正式解码之前，先预热解码的音频数量，一般刚启动时第一次解码不能实时，5s音频预计需要20s才能解码完。
CUDA_VISIBLE_DEVICES="0" ./build/bin/decoder_main \
    --warmup $warmup   \
    --thread_num $thread_num   \
    --frame_shift 100 \
    --chunk_size 16 ${wfst_decode_opts} ${decode_opts} \
    --ctc_weight 0.5 --rescoring_weight 1.0 \
    --reverse_weight 0.3 \
    --min_trailing_silence $min_trailing_silence  \
    --max_utterance_length $max_utterance_length  \
    --model_path $model_path \
    --unit_path $unit_path  \
    --wav_scp $wav_scp \
    --result $decode_result 2>&1 | tee local_multithread.log

#输出解码实时率信息，超过1表示不可实时
tail local_multithread.log | grep RTF

#计算解码结果错误率
if [ -f $label_file ];then
python3 ../resource/tools/compute-wer.py --char=1 --v=1 \
  $label_file $decode_result > $wer_result
tail $wer_result | grep -C 2 "Overall"
fi