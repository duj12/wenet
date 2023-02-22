#!/bin/bash

. ./path.sh || exit 1;

tmp_fifofile="/tmp/$$.fifo"
mkfifo $tmp_fifofile   # 新建一个FIFO类型的文件
exec 6<>$tmp_fifofile  # 将FD6指向FIFO类型
rm $tmp_fifofile  #删也可以，

thread_num=16  # 定义最大线程数

#根据线程总数量设置令牌个数
#事实上就是在fd6中放置了$thread_num个回车符
for ((i=0;i<${thread_num};i++));do
    echo
done >&6

# This bpe model is trained on ESPnet ASR xmov_cs/asr2 training data set.
# ALL paths are relative path
bpecode=conf/zh6300char_en5700bpe.model
unit_file=data/dict_bpe/tokens.txt          # the AM's model unit, which is corresponding to bpecode
original_lexicon=data/lexicon/lexicon.txt   # the original lexicon, may with pronunciation
original_vocab=data/lexicon/vocab.txt       # the original vocab list
lm_corpus_paths=data/lm_corpus/train.list    # the sub-text's path list
lm_test_paths=data/lm_corpus/test.list


order=6          #the grade of n-gram. 250G corpus 5-gram consume about 680G memory.
prune=0 #0.000000001   #prune
chinese_unit=chars
#give lm a specific name, in case you train many different LMs
LM_name=  #"lm_250G_${order}gram+asrtext_6gram_"$chinese_unit


. tools/parse_options.sh || exit 1;


stage=$1
stop_stage=$2
echo "LM_name is $LM_name"

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

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] ; then
  echo
  echo "0. Prepare text: split the chinese text into chars/words."
  echo
  cut -d' ' -f1 $original_lexicon | uniq > $original_vocab

  if [ $chinese_unit = "chars" ] ; then
    echo "split Chinese text into ${chinese_unit}."
    for path_list in $lm_test_paths  $lm_corpus_paths ; do
      for text_path in `cat $path_list` ; do
        read -u6
        {
        new_text_path=$text_path.$chinese_unit
        local/split_${chinese_unit}.py $text_path $new_text_path $original_vocab
        echo >&6 # 当进程结束以后，再向FD6中加上一个回车符，即补上了read -u6减去的那个
        echo $new_text_path
        } &
      done > $path_list.$chinese_unit
      wait   # wait for all threads finish
    done

  elif [ $chinese_unit = "words" ]; then
    # 这里提前将friso安装到tools路径下，并且修改tst_friso.c为获取输入文件，分词写入到输出文件的格式，编译并配置好词典路径
    echo "split Chinese text into ${chinese_unit}."
    cp $original_vocab tools/friso/dict/UTF-8/
    #process train corpus, in batch(multi-job)
    for path_list in $lm_test_paths  $lm_corpus_paths ; do
      for text_path in `cat $path_list` ; do
        read -u6
        {
        new_text_path=$text_path.$chinese_unit
        tools/friso/src/friso -init tools/friso/friso.ini $text_path $new_text_path >&2 #将分词产生的日志定向到标准错误
        echo >&6 # 当进程结束以后，再向FD6中加上一个回车符，即补上了read -u6减去的那个
        echo $new_text_path
        } &
      done > $path_list.$chinese_unit
      wait   # wait for all threads finish
    done
  fi
fi
lm_test_paths=$lm_test_paths.$chinese_unit
lm_corpus_paths=$lm_corpus_paths.$chinese_unit

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo
  echo "1. Prepare dict: convert each word(for chinese we use chars/words) into bpe units."
  echo
  chinese_bpe_is_char=1    # the bpe model is trained with chinese chars and english bpe
  if [ $chinese_unit = "chars" ]; then
    chinese_bpe_is_char=0  # 设为0确保将除了bpe_vocab中的单个汉字之外的词排除出词表之外
  fi
  tools/fst/prepare_dict.py $unit_file $original_lexicon \
    $converted_lexicon ${bpecode} ${chinese_bpe_is_char}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo
  echo "2. Train lm, in arpa format, may take such a long time..."
  echo
  cut -d' ' -f1 $converted_lexicon > $word_vocab
  local/train_large_lm.sh $lm_corpus_paths $word_vocab $order $prune \
  $lm_path $thread_num  $tmp_path
  #ppl test
  echo
  echo "Test the lm.arpa in ${lm_path} in file list ${lm_test_paths}."
  for test_set_path in `cat $lm_test_paths`; do
    ngram -ppl ${test_set_path} -order $order -lm $lm_path/lm.arpa  #-debug 2
  done
  echo
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo
  echo "3. Build decoding TLG, convert arpa to fst"
  echo
  tools/fst/compile_lexicon_token_fst.sh  $dict_path $tmp_path $lang_path
  tools/fst/make_tlg.sh $lm_path $lang_path $fst_path || exit 1;
fi

exec 6>&- # 关闭FD6
echo "over" # 表示脚本运行结束