
# WeNet Python Binding

This is a python binding of WeNet.

WeNet is a production first and production ready end-to-end speech recognition toolkit.

The best things of the binding are:

1. Multiple languages supports, including English, Chinese. Other languages are in development.
2. Non-streaming and streaming API
3. N-best, contextual biasing, and timestamp supports, which are very important for speech productions.
4. Alignment support. You can get phone level alignments this tool, on developing.

# 使用说明

## 安装方法

由于这个版本是不同于WeNet官方的版本，内部很多代码做过更改，因此不能直接使用官方的包。

需要在本地克隆此代码仓库，然后进行编译

如果编译GPU版本，则运行
```shell
cd asr-online/binding/python
python setup.py install
```      
    
如需编译CPU版本，需要预先安装torch-cpu=1.13版本。运行如下代码：
```shell
cd asr-online/binding/python
pip --no-cache-dir install torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
python setup_cpu.py install
```
    
## 接口修改与绑定
    
    目前已经开放的接口见api/wenet_api.cc，可以根据需要进行二次开发。
    将接口绑定到python方法的具体代码可参阅cpp/bindding.cc。
    同时，需要在py/decoder.py里将做过的改动进行同步。
    


## 使用示例
具体调用方式可以参考run_demo.py, 这里给出一个简单的流式调用例子：

``` python
import sys
import torch
import wave
# 导入所需的模型，和解码器模块
from wenetruntime.decoder import ASRModel, Decoder

test_wav = sys.argv[1]   # 测试音频路径
model_dir = sys.argv[2]  # 模型路径

# 首先加载模型
model = ASRModel(model_dir, num_thread = 1)   

# 将模型传给解码器， 这里除了模型之外的其他参数都使用默认参数
decoder = Decoder(model, 
                 lang='chs',
                 nbest= 1,
                 enable_timestamp = False,
                 enable_itn = False,
                 context = None,
                 context_score = 3.0,
                 continuous_decoding = False,
                 vad_trailing_silence = 1000, 
                 fbank_frame_shift = 160)

# 读取音频，目前只支持单通道16k采样率16bit的音频
with wave.open(test_wav, 'rb') as fin:
    assert fin.getnchannels() == 1
    wav = fin.readframes(fin.getnframes())
interval = int(0.3 * 16000) * 2
# 模拟音频流式输入，每个chunk为0.3s，进行解码
for i in range(0, len(wav), interval):
    last = False if i + interval < len(wav) else True
    chunk_wav = wav[i: min(i + interval, len(wav))]
    ans = decoder.decode(chunk_wav, last)
    print(ans)
```

`ASRModel` 的参数为模型路径，以及使用的CPU核数（可使用全部CPU核）
* `model_dir` (str): is the `Runtime Model` directory, it contains the following files.
  
* 必备资源
  * `final.zip`: TorchScript ASR model. ASR模型，TorchScript格式。
  * `units.txt`: modeling units file. ASR模型的建模单元。
  
* 可选资源
  * `TLG.fst`: optional, it means decoding with LM when `TLG.fst` is given. N-gram语言模型
  * `words.txt`: optional, word level symbol table for decoding with `TLG.fst`. 语言模型对应的词表。
  * `hotwords.txt`: optional,热词列表，一个词一行。
  * `zh_itn_tagger.fst`:optional, 用于文本反正则。
  * `zh_itn_verbalizer.fst`:optional, 用于文本反正则。


`Decoder`中的其他参数信息和推荐配置如下

* `lang` (str): The language you used, `chs` for Chinese, and `en` for English. 使用默认值chs即可，不需要更改
* `nbest` (int): Output the top-n best result. 支持最多10个候选结果，一般设置为1。
* `enable_timestamp` (bool): Whether to enable the word level timestamp. 需要时间戳时设为True， 否则设置为False。中文返回每个汉字的时间戳，英文则是返回BPE单元的时间戳。
* `enable_itn` (bool): Whether to enable Inverse Text Normalization. 需要文本反正则时设为True，否则设置为False。设为True时会将中文汉字转成可能的阿拉伯数字形式。
* `context` (List[str]): a list of context biasing words. 如果有热词，则传入热词list
* `context_score` (float): context bonus score. 热词增强的分数，一般设置为3.0不需要改，此值越大，则热词越容易被激活。
* `continuous_decoding` (bool): Whether to enable continuous(long) decoding. 是否使用ASR内置模块断句进行连续解码。V1.4.0之后的版本，推荐设置为False。如果此值设置为False，则是靠外置VAD模块进行断句；如果设为True，则是使用ASR内部模块进行断句。
* `vad_trailing_silence` (int): the trailing silence length of VAD, in ms. VAD拖尾静音参数，单位为毫秒，一般设置为1000，表示语音之后静音超过1000ms，则音频流被截断，解码器重置，后续音频流解码不依赖此前历史信息。此值越大，则音频流越不容易被切断。
* `fbank_frame_shift` (int): the frame shift, in sample counts. 提取特征时每一帧的移动间隔，单位为采样点数，推荐设置为160。V1.4.0之后提供的模型都是帧移为160，对应帧率为25。


Note:

1. For macOS, wenetruntime packed `libtorch.so`, so we can't import torch and wenetruntime at the same time.
2. For Windows and Linux, wenetruntime depends on torch. Please install and import the same version `torch` as wenetruntime.

