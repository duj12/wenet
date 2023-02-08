## Make BIG n-gram LM

细节可以参考http://fancyerii.github.io/dev287x/lm/

一、小数据

假设有去除特殊符号的训练文本trainfile.txt，以及测试文本testfile.txt，那么训练一个语言模型以及对其进行评测的步骤如下：

1：词频统计

      ngram-count -text trainfile.txt -order 3 -write trainfile.count

      其中-order 3为3-gram，trainfile.count为统计词频的文本

2：模型训练

      ngram-count -read trainfile.count -order 3 -lm trainfile.lm  -interpolate -kndiscount

      其中trainfile.lm为生成的语言模型，-interpolate和-kndiscount为插值与折回参数

3：测试（困惑度计算）

     ngram -ppl testfile.txt -order 3 -lm trainfile.lm -debug 2 > file.ppl

     其中testfile.txt为测试文本，-debug 2为对每一行进行困惑度计算，类似还有-debug 0 , -debug 1, -debug 3等，最后  将困惑度的结果输出到file.ppl

二、对于大文本的语言模型训练不能使用上面的方法，主要思想是将文本切分，分别计算，然后合并。步骤如下：

1：切分数据

      split -l 10000 trainfile.txt filedir/

      即每10000行数据为一个新文本存到filedir目录下。

2：对每个文本统计词频

      make-batch-counts filepath.txt 1 cat ./counts -order 3

      其中filepath.txt为切分文件的全路径，可以用命令实现：ls $(echo $PWD)/* > filepath.txt，将统计的词频结果存放在counts目录下

3：合并counts文本并压缩

      merge-batch-counts ./counts

      不解释

4：训练语言模型

      make-big-lm -read ../counts/*.ngrams.gz -lm ../split.lm -order 3

     用法同ngram-counts

5: 测评（计算困惑度）

    ngram -ppl filepath.txt -order 3 -lm split.lm -debug 2 > file.ppl

6: 领域数据插值
    
首先计算插值系数

    ngram -debug 2 -order 3 -lm general.ngram.gz -ppl data/dev.txt > lm1.ppl
    ngram -debug 2 -order 3 -lm domain.ngram.gz -ppl data/dev.txt > lm2.ppl
    compute-best-mix lm*.ppl > best-mix.ppl
    
然后进行插值，并且限定词典

    ngram -debug 1 -order 3 -lm domain.ngram.gz -lambda 0.8 -mix-lm general.ngram.gz \
        -write-lm general+domain.ngram.gz -vocab data/dict.vocab -limit-vocab

7: 模型裁剪

    ngram -debug 1 -lm general.ngram.gz -prune 1e-9 -write-lm general-prune1e-9.ngram.gz

关于裁剪的参数，有如下实验结果

    prune值	10−5	10−6	10−7	10−8	10−9	10−10	未裁剪
    模型大小	336K	2.1M	12M	61M	205M	263M	286M
    PPL	258	178	136	118	113.49	113.21	113.19

