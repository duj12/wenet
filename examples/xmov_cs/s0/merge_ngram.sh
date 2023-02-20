#!/bin/bash

. ./path.sh || exit 1;

general_ngram=data/lm_250G_5gram_chars/local/lm/lm.arpa
domain_ngram=data/lm_asrtext_6gram_chars/local/lm/lm.arpa

domain_text=data/train_xmov1/lm.txt

dict=data/lm_asrtext_6gram_chars/local/dict/word.vocab
merged_ngram=data/lm_250G_5gram+asrtext_6gram_chars/local/lm/lm.arpa

dev_text=data/lm_dev.txt
prune=0

merged_dir=`dirname $merged_ngram`
merged_name=`basename $merged_ngram`
mkdir -p $merged_dir

stage=0

#对通用语言模型进行裁剪
if [ $stage -le -2 ]; then
  general_ngram_prune=1e-9
  if [ ! -z $general_ngram_prune ]; then
    prune_dir=`dirname $general_ngram`
    prune_name=lm_prune${prune}.arpa
    ngram -debug 1 -lm $general_ngram -prune $prune -write-lm $prune_dir/$prune_name
    general_ngram=$prune_dir/$prune_name
  fi
fi

#然后生成垂域语言模型
if [ $stage -le -1 ]; then
  if [ ! -f $domain_ngram ] && [ -f $domain_text ] ; then


  fi
fi

if [ $stage -le 0 ]; then
ngram -debug 2 -order 6 -lm $general_ngram -ppl $dev_text > $merged_dir/lm1.ppl
ngram -debug 2 -order 6 -lm $domain_ngram -ppl $dev_text > $merged_dir/lm2.ppl
compute-best-mix $merged_dir/lm1.ppl $merged_dir/lm2.ppl > $merged_dir/best-mix.ppl
fi

if [ $stage -le 1 ]; then
lambda=`tail -n 1 $merged_dir/best-mix.ppl | awk '{print $NF}' | awk -F")" '{print $1}'`
echo "best mix lambda = $lambda"
ngram -debug 1 -order 6 -lm $domain_ngram -lambda $lambda -mix-lm $general_ngram \
    -write-lm $merged_ngram -vocab $dict -limit-vocab
fi


if [ $stage -le 2 ]; then
if [ $prune -gt 0 ] ; then
  ngram -debug 1 -lm $merged_ngram -prune $prune -write-lm  $merged_dir/prune{prune}_$merged_name
  ngram -debug 2 -order 6 -lm $merged_dir/prune{prune}_$merged_name -ppl $dev_text > $merged_dir/lm_merge.ppl
else
  ngram -debug 2 -order 6 -lm $merged_ngram -ppl $dev_text > $merged_dir/lm_merge.ppl
fi
fi
echo "over" # 表示脚本运行结束