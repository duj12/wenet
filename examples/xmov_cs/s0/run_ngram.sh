#!/bin/bash

. ./path.sh || exit 1;


stage=1 # start from 0 if you need to start from data preparation
stop_stage=3

. tools/parse_options.sh || exit 1;

# This bpe model is trained on ESPnet ASR xmov_cs/asr2 training data set.
bpecode=conf/zh6300char_en5700bpe.model
unit_file=data/dict_bpe/tokens.txt          # the AM's model unit, which is corresponding to bpecode
original_lexicon=data/lexicon/lexicon.txt   # the original dict
lm_corpus_paths=data/lm_corpus/train.list    # the sub-text's path list
lm_test_paths=data/lm_corpus/test.list

order=4          #the grade of n-gram
prune=0.000000001   #prune 10e-9
LM_name="LM_250G"   #give lm a specific name, in case you train many different lms

dict_path=data/$LM_name/local/dict
mkdir -p $dict_path
cp $unit_file $dict_path/units.txt
converted_lexicon=$dict_path/lexicon.txt  #the converted dict
word_vocab=$dict_path/word.vocab


lm_path=data/$LM_name/local/lm   # the dir to store lm.arpa
mkdir -p $lm_path

# some dir needed when compile fst
tmp_path=data/$LM_name/local/tmp
lang_path=data/$LM_name/local/lang
fst_path=data/$LM_name/lang_test

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo
  echo "1. Prepare dict: convert each word(for chinese we use chars/words) into bpe units."
  echo
  tools/fst/prepare_dict.py $unit_file $original_lexicon \
    $converted_lexicon ${bpecode}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo
  echo "2. Train lm, in arpa format, may take such a long time..."
  echo
  cut -d' ' -f1 $converted_lexicon > $word_vocab
  local/train_large_lm.sh $lm_corpus_paths $word_vocab $order $prune $lm_path
  #ppl test
  echo
  echo "Test the lm.arpa in ${lm_path} in file list ${lm_test_paths}."
  ngram -ppl $lm_test_paths -order $order -lm $lm_path/lm.arpa -debug 2
  echo
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo
  echo "3. Build decoding TLG, convert arpa to fst"
  echo
  tools/fst/compile_lexicon_token_fst.sh  $dict_path $tmp_path $lang_path
  tools/fst/make_tlg.sh $lm_path $lang_path $fst_path || exit 1;
fi
