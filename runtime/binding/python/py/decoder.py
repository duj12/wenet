# Copyright (c) 2022  Binbin Zhang(binbzha@qq.com)
#               2023  dujing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import _wenet
from .vad import CRNN_VAD_STREAM
import json, struct

class ASRModel:
    def __init__(self, model_dir: str,
                 num_thread: int = 1):
        """
        :param model_dir: 模型资源的路径，必须包含final.zip和units.txt，可选包含TLG.fst,words.txt,hot_words.txt
        num_thread: 默认每路解码所需的线程数量，也可以通过在启动Decode时传入环境变量OMP_NUM_THREADS=num_thread进行设置
        """
        _wenet.wenet_set_log_level(2)
        self.model = _wenet.wenet_init_resource(model_dir, num_thread)

    def __del__(self):
        _wenet.wenet_free_resource(self.model)

    def set_log_level(self, level: int = 2):
        """
        :param level: 0,1,2. normally increase to 2. higher with more log information
        :return:
        """
        _wenet.wenet_set_log_level(level)  # set the log level

class Decoder:

    def __init__(self,
                 resource: ASRModel,
                 lang: str = 'chs',
                 nbest: int = 1,
                 enable_timestamp: bool = False,
                 enable_itn: bool = False,
                 context: Optional[List[str]] = None,
                 context_score: float = 3.0,
                 continuous_decoding: bool = False,
                 vad_trailing_silence: int = 1000, 
                 fbank_frame_shift: int = 160, ):
        """ Init WeNet decoder
        Args:
            model: The ASR model resources
            lang: language type of the model
            nbest: nbest number for the final result
            enable_timestamp: whether to enable word level timestamp
            context: context words
            context_score: bonus score when the context is matched
            continuous_decoding: enable countinous decoding or not
            vad_trailing_silence: the silence length in ms.
                If the silence is longer than this, the audio will be cutted.
            fbank_frame_shift: the frame shift in feature extraction. 
        """
        self.d = _wenet.wenet_init()
        _wenet.wenet_set_log_level(0)  # set the log level
        self.load_resource(resource.model)
        self.set_language(lang)
        self.set_nbest(nbest)
        self.enable_timestamp(enable_timestamp)
        self.set_itn(enable_itn)
        if context is not None:
            self.add_context(context)
            self.set_context_score(context_score)

        self.vad = None
        if not continuous_decoding:
            self.vad = CRNN_VAD_STREAM(
                max_trailing_silence= vad_trailing_silence)

        self.set_continuous_decoding(continuous_decoding)
        self.set_vad_trailing_silence(vad_trailing_silence)
        self.set_frame_shift(fbank_frame_shift)

        # some private members, for warmup telling
        self._is_first_decoded = False
        self._is_warmed = False

    def __del__(self):
        _wenet.wenet_free(self.d)

    def __version__(self):
        return "1.4.1"

    def set_log_level(self, level):
        """
        :param level: 0,1,2.... normally increase to 2. higher with more log information
        :return:
        """
        _wenet.wenet_set_log_level(level)  # set the log level

    def is_warmed(self):
        return self._is_warmed

    def load_resource(self, model):
        _wenet.wenet_set_decoder_resource(self.d, model)

    def reset(self):
        """ Reset status for next decoding """
        _wenet.wenet_reset(self.d)
        if not self.vad is None:
            self.vad.reset_state()

    def init_decoder(self):
        """Init the user specific decoder"""
        _wenet.wenet_init_decoder(self.d)

    def reset_user_decoder(self):
        """Init the user specific decoder"""
        _wenet.wenet_reset_user_decoder(self.d)

    def set_nbest(self, n: int):
        assert n >= 1
        assert n <= 10
        _wenet.wenet_set_nbest(self.d, n)

    def enable_timestamp(self, flag: bool):
        tag = 1 if flag else 0
        _wenet.wenet_set_timestamp(self.d, tag)

    def set_itn(self, flag: bool):
        tag = 1 if flag else 0
        _wenet.wenet_set_itn(self.d, tag)

    def add_context(self, contexts: List[str]):
        """add common hotwords"""
        for c in contexts:
            assert isinstance(c, str)
            _wenet.wenet_add_context(self.d, c)

    def set_context_score(self, score: float):
        _wenet.wenet_set_context_score(self.d, score)

    def reset_user_context(self, user_contexts: List[str]):
        """reset the customized hotwords"""
        _wenet.wenet_clear_user_context(self.d)
        for c in user_contexts:
            assert isinstance(c, str)
            _wenet.wenet_add_user_context(self.d, c)

    def set_language(self, lang: str):
        assert lang in ['chs', 'en']
        _wenet.wenet_set_language(self.d, lang)

    def set_continuous_decoding(self, continuous_decoding: bool):
        flag = 1 if continuous_decoding else 0
        _wenet.wenet_set_continuous_decoding(self.d, flag)

    def set_vad_trailing_silence(self, vad_trailing_silence: int):
        _wenet.wenet_set_vad_trailing_silence(self.d, vad_trailing_silence)
        if not self.vad is None:
            self.vad.max_trailing_silence = vad_trailing_silence

    def set_frame_shift(self, frame_shift: int):
        _wenet.wenet_set_frame_shift(self.d, frame_shift)

    def decode(self, pcm: bytes, last: bool = True) -> str:
        """ Decode the input data

        Args:
            pcm: wav pcm
            last: if it is the last package of the data
        """
        assert isinstance(pcm, bytes)

        vad_state = 0
        # chunk is better to be longer than 0.2 second, for better vad performance
        if not last and not self.vad is None and len(pcm)>=6400:
            # convert bytes into float32
            data = []
            for i in range(0, len(pcm), 2):
                value = struct.unpack('<h', pcm[i:i + 2])[0]
                data.append(value / 32768.0)
            *_, vad_state = self.vad.stream_asr_endpoint(data)

        finish = 1 if (last or vad_state > 0) else 0
        _wenet.wenet_decode(self.d, pcm, len(pcm), finish)
        result = _wenet.wenet_get_result(self.d)
        if finish:  # Reset status for next decoding automatically
            self.reset()
            if self._is_first_decoded == True:
                self._is_warmed = True
            self._is_first_decoded = True

        if not self.vad is None:
            result = json.loads(result) if len(result) > 0 else None
            if result and "vad_state" in result:
                result["vad_state"] = vad_state
            if vad_state == 1: # noise, sentence in result should be clear
                if result and "nbest" in result:
                    result["nbest"] = []
            result = json.dumps(result) if result else ""
        return result

    def decode_wav(self, wav_file: str) -> str:
        """ Decode wav file, we only support:
            1. 16k sample rate
            2. mono channel
            3. sample widths is 16 bits / 2 bytes
        """
        import wave
        with wave.open(wav_file, 'rb') as fin:
            assert fin.getnchannels() == 1
            assert fin.getsampwidth() == 2
            assert fin.getframerate() == 16000
            wav = fin.readframes(fin.getnframes())
        return self.decode(wav, True)
