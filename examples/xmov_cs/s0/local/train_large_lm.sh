#!/bin/bash
stage=1
stop_stage=3

# To be run from one directory above this script.
. ./path.sh

text=$1  # text file paths, split the large corpus to sub-text files.
lexicon=$2  # lexicon: split each word to bpe units.
order=$3
prune=$4
dir=$5
#text_with_uttname=$6
mkdir -p $dir

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

# Check SRILM tools
if ! which ngram-count > /dev/null; then
    echo "srilm tools are not found, please download it and install it from: "
    echo "http://www.speech.sri.com/projects/srilm/download.html"
    echo "Then add the tools to your PATH"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#2：对每个文本统计词频，将统计的词频结果存放在counts目录下
echo "Get word counts frequency of each file in ${text}"
##if [[ $text_with_uttname -eq 1 ]]; then
##use the following following function to remove the utt-name in the text files
##textfilter="while read line; do echo line | cut -d' ' -f2- ; done  |"
#make-batch-counts-with-uttname $text 1 cat $dir/counts -order $order
##其中filepath.txt为切分文件的全路径，可以用命令实现：ls $(echo $PWD)/* > filepath.txt
#elif [[ $text_with_uttname -eq 0 ]]; then
  make-batch-counts $text 1 cat $dir/counts -order $order
#fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#3：合并counts文本并压缩, 生成.ngram.gz后缀的文件
echo "Merging the word frequency counts."
merge-batch-counts $dir/counts
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
#4：训练语言模型
echo "Making BIG LM, with order=${order}, prune=${prune}, vocab=${lexicon}"

  make-big-lm -read $dir/counts/*.ngrams.gz -order $order -limit-vocab -vocab $lexicon -unk \
    -map-unk "<UNK>" -kndiscount  -interpolate -prune $prune -lm $dir/lm.arpa

#用法同ngram-counts
fi


