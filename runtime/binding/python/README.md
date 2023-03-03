# 使用说明

## 安装方法

    由于这个版本是不同于WeNet官方的版本，内部很多代码做过更改，因此不能直接使用官方的包。

    需要在本地克隆此代码仓库，然后运行
    cd asr-online/binding/python
    python setup.py install  
    
    默认的setup.py使用CPU编译，如需编译GPU版本，则需要在cmake编译选项中添加"-DGPU=ON"。
    如果遇到torch安装失败的问题，可以考虑切换源，或者提前安装上torch，再运行setup.py安装
    推荐使用torch==1.11.0(或1.11.0+cu113), python==3.7或3.8, 经过测试此环境能正常运行。
    
## 接口修改与绑定
    
    目前已经开放的接口见api/wenet_api.cc，可以根据需要进行二次开发。
    将接口绑定到python方法的具体代码可参阅cpp/bindding.cc。
    同时，需要在py/decoder.py里将做过的改动进行同步。
    

# WeNet Python Binding

This is a python binding of WeNet.

WeNet is a production first and production ready end-to-end speech recognition toolkit.

The best things of the binding are:

1. Multiple languages supports, including English, Chinese. Other languages are in development.
2. Non-streaming and streaming API
3. N-best, contextual biasing, and timestamp supports, which are very important for speech productions.
4. Alignment support. You can get phone level alignments this tool, on developing.


## Usage

Note:

1. For macOS, wenetruntime packed `libtorch.so`, so we can't import torch and wenetruntime at the same time.
2. For Windows and Linux, wenetruntime depends on torch. Please install and import the same version `torch` as wenetruntime.

### Non-streaming Usage

``` python
import sys
import torch
import wenetruntime as wenet

wav_file = sys.argv[1]
decoder = wenet.Decoder(lang='chs')
ans = decoder.decode_wav(wav_file)
print(ans)
```

You can also specify the following parameter in `wenet.Decoder`

* `lang` (str): The language you used, `chs` for Chinese, and `en` for English.
* `model_dir` (str): is the `Runtime Model` directory, it contains the following files.
   If not provided, official model for specific `lang` will be downloaded automatically.

  * `final.zip`: runtime TorchScript ASR model.
  * `units.txt`: modeling units file
  * `TLG.fst`: optional, it means decoding with LM when `TLG.fst` is given.
  * `words.txt`: optional, word level symbol table for decoding with `TLG.fst`

  Please refer https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md for the details of `Runtime Model`.

* `nbest` (int): Output the top-n best result.
* `enable_timestamp` (bool): Whether to enable the word level timestamp.
* `context` (List[str]): a list of context biasing words.
* `context_score` (float): context bonus score.
* `continuous_decoding` (bool): Whether to enable continuous(long) decoding.

For example:
``` python
decoder = wenet.Decoder(model_dir,
                        lang='chs',
                        nbest=5,
                        enable_timestamp=True,
                        context=['不忘初心', '牢记使命'],
                        context_score=3.0)
```

### Streaming Usage

``` python
import sys
import torch
import wave
import wenetruntime as wenet

test_wav = sys.argv[1]

with wave.open(test_wav, 'rb') as fin:
    assert fin.getnchannels() == 1
    wav = fin.readframes(fin.getnframes())

decoder = wenet.Decoder(lang='chs')
# We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
interval = int(0.5 * 16000) * 2
for i in range(0, len(wav), interval):
    last = False if i + interval < len(wav) else True
    chunk_wav = wav[i: min(i + interval, len(wav))]
    ans = decoder.decode(chunk_wav, last)
    print(ans)
```

You can use the same parameters as we introduced above to control the behavior of `wenet.Decoder`

