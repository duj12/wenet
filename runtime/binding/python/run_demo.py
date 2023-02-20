import sys
import torch
import wave
import wenetruntime as wenet

test_wav = sys.argv[1]

with wave.open(test_wav, 'rb') as fin:
    assert fin.getnchannels() == 1
    wav = fin.readframes(fin.getnframes())

# demo中使用自研的模型，同时开放了VAD拖尾静音长度的参数（默认是1000ms）
# 对于流式场景，需要将continuous_decoding_设置为true
# 关于其他decoder的可设置参数，见py/decoder.py
decoder = wenet.Decoder(model_dir='../../resource/ASR/', lang='chs',
                        continuous_decoding=True,
                        vad_trailing_silence=500)
# We suppose the wav is 16k, 16bits, and decode every 0.5 seconds
interval = int(0.5 * 16000) * 2
for i in range(0, len(wav), interval):
    last = False if i + interval < len(wav) else True
    chunk_wav = wav[i: min(i + interval, len(wav))]
    ans = decoder.decode(chunk_wav, last)
    print(ans)