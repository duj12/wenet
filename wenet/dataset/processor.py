# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
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

import logging
import json
import random
import re
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.nn.utils.rnn import pad_sequence

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))


def tar_file_and_group(data):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix == 'txt':
                        example['txt'] = file_obj.read().decode('utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data):
    """ Parse key/wav/txt from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/txt

        Returns:
            Iterable[{key, wav, txt, sample_rate}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'txt' in obj
        key = obj['key']
        wav_file = obj['wav']
        txt = obj['txt']
        try:
            if 'start' in obj:
                assert 'end' in obj
                sample_rate = torchaudio.backend.sox_io_backend.info(
                    wav_file).sample_rate
                start_frame = int(obj['start'] * sample_rate)
                end_frame = int(obj['end'] * sample_rate)
                waveform, _ = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_file,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                waveform, sample_rate = torchaudio.load(wav_file)
            example = dict(key=key,
                           txt=txt,
                           wav=waveform,
                           sample_rate=sample_rate)
            yield example
        except Exception as ex:
            logging.warning('Failed to read {}'.format(wav_file))


def filter(data,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1):
    """ Filter sample according to feature and label length
        Inplace operation.

        Args::
            data: Iterable[{key, wav, label, sample_rate}]
            max_length: drop utterance which is greater than max_length(10ms)
            min_length: drop utterance which is less than min_length(10ms)
            token_max_length: drop utterance which is greater than
                token_max_length, especially when use char unit for
                english modeling
            token_min_length: drop utterance which is
                less than token_max_length
            min_output_input_ratio: minimal ration of
                token_length / feats_length(10ms)
            max_output_input_ratio: maximum ration of
                token_length / feats_length(10ms)

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'label' in sample
        # sample['wav'] is torch.Tensor, we have 100 frames every second
        num_frames = sample['wav'].size(1) / sample['sample_rate'] * 100
        if num_frames < min_length:
            continue
        if num_frames > max_length:
            continue
        if len(sample['label']) < token_min_length:
            continue
        if len(sample['label']) > token_max_length:
            continue
        if num_frames != 0:
            if len(sample['label']) / num_frames < min_output_input_ratio:
                continue
            if len(sample['label']) / num_frames > max_output_input_ratio:
                continue
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, speeds=None):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            speeds(List[float]): optional speed

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed = random.choice(speeds)
        if speed != 1.0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
            sample['wav'] = wav

        yield sample

def add_noise_rir(data,
                  rir_scp = "./data/rir.list",
                  total_rir = 417,
                  rir_apply_prob = 0.2,
                  noise_scp="./data/noise.list",
                  total_noise = 2556,
                  noise_apply_prob=0.8,
                  noise_db_low = 5,
                  noise_db_high = 15, ):
    import numpy as np
    import soundfile, scipy.signal
    import linecache
    def framing(
            x,
            frame_length: int = 512,
            frame_shift: int = 256,
            centered: bool = True,
            padded: bool = True,
    ):
        if x.size == 0:
            raise ValueError("Input array size is zero")
        if frame_length < 1:
            raise ValueError("frame_length must be a positive integer")
        if frame_length > x.shape[-1]:
            raise ValueError("frame_length is greater than input length")
        if 0 >= frame_shift:
            raise ValueError("frame_shift must be greater than 0")

        if centered:
            pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
                (frame_length // 2, frame_length // 2)
            ]
            x = np.pad(x, pad_shape, mode="constant", constant_values=0)

        if padded:
            # Pad to integer number of windowed segments
            # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
            #  with integer nseg
            nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
            pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
            x = np.pad(x, pad_shape, mode="constant", constant_values=0)

        # Created strided array of data segments
        if frame_length == 1 and frame_length == frame_shift:
            result = x[..., None]
        else:
            shape = x.shape[:-1] + (
                (x.shape[-1] - frame_length) // frame_shift + 1,
                frame_length,
            )
            strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
            result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return result
    def detect_non_silence(
            x: np.ndarray,
            threshold: float = 0.01,
            frame_length: int = 1024,
            frame_shift: int = 512,
            window: str = "boxcar",
    ) -> np.ndarray:
        """Power based voice activity detection.
        Args:
            x: (Channel, Time)
        >>> x = np.random.randn(1000)
        >>> detect = detect_non_silence(x)
        >>> assert x.shape == detect.shape
        >>> assert detect.dtype == np.bool
        """
        if x.shape[-1] < frame_length:
            return np.full(x.shape, fill_value=True, dtype=np.bool)

        if x.dtype.kind == "i":
            x = x.astype(np.float64)
        # framed_w: (C, T, F)
        framed_w = framing(
            x,
            frame_length=frame_length,
            frame_shift=frame_shift,
            centered=False,
            padded=True,
        )
        framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
        # power: (C, T)
        power = (framed_w ** 2).mean(axis=-1)
        # mean_power: (C, 1)
        mean_power = np.mean(power, axis=-1, keepdims=True)
        if np.all(mean_power == 0):
            return np.full(x.shape, fill_value=True, dtype=np.bool)
        # detect_frames: (C, T)
        detect_frames = power / mean_power > threshold
        # detects: (C, T, F)
        detects = np.broadcast_to(
            detect_frames[..., None], detect_frames.shape + (frame_shift,)
        )
        # detects: (C, TF)
        detects = detects.reshape(*detect_frames.shape[:-1], -1)
        # detects: (C, TF)
        return np.pad(
            detects,
            [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
            mode="edge",
        )

    def _convolve_rir(rir_scp, total_rir, speech, power):
        rir_path = linecache.getline(rir_scp, random.randrange(1, total_rir)).strip()
        rir = None
        if rir_path is not None and isinstance(rir_path, str) and len(rir_path)>0:
            rir, _ = soundfile.read(rir_path, dtype=np.float64, always_2d=True)

            # rir: (Nmic, Time)
            rir = rir.T

            # speech: (Nmic, Time)
            # Note that this operation doesn't change the signal length
            speech = scipy.signal.convolve(speech, rir, mode="full")[
                     :, : speech.shape[1]
                     ]
            # Reverse mean power to the original power
            power2 = (speech[detect_non_silence(speech)] ** 2).mean()
            speech = np.sqrt(power / max(power2, 1e-10)) * speech
        return speech, rir

    def _add_noise(noise_scp, total_noise, speech, power, noise_db_low, noise_db_high):
        nsamples = speech.shape[1]
        noise_path = linecache.getline(noise_scp, random.randrange(1, total_noise)).strip()
        noise = None
        if noise_path is not None and isinstance(noise_path, str) and len(noise_path)>0:
            noise_db = np.random.uniform(noise_db_low, noise_db_high)
            with soundfile.SoundFile(noise_path) as f:
                if f.frames == nsamples:
                    noise = f.read(dtype=np.float64, always_2d=True)
                elif f.frames < nsamples:
                    # if f.frames / nsamples < self.short_noise_thres:
                    #     logging.warning(
                    #         f"Noise ({f.frames}) is much shorter than "
                    #         f"speech ({nsamples}) in dynamic mixing"
                    #     )
                    offset = np.random.randint(0, nsamples - f.frames)
                    # noise: (Time, Nmic)
                    noise = f.read(dtype=np.float64, always_2d=True)
                    # Repeat noise
                    noise = np.pad(
                        noise,
                        [(offset, nsamples - f.frames - offset), (0, 0)],
                        mode="wrap",
                    )
                else:
                    offset = np.random.randint(0, f.frames - nsamples)
                    f.seek(offset)
                    # noise: (Time, Nmic)
                    noise = f.read(nsamples, dtype=np.float64, always_2d=True)
                    if len(noise) != nsamples:
                        raise RuntimeError(f"Something wrong: {noise_path}")
            # noise: (Nmic, Time)
            noise = noise.T

            noise_power = (noise ** 2).mean()
            scale = (
                    10 ** (-noise_db / 20)
                    * np.sqrt(power)
                    / np.sqrt(max(noise_power, 1e-10))
            )
            speech = speech + scale * noise
        return speech, noise

    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        speech = sample['wav'].cpu().numpy() if sample['wav'].is_cuda else sample["wav"].numpy()
        # speech: (Nmic, Time)
        if speech.ndim == 1:
            speech = speech[None, :]

        # Calc power on non silence region
        power = (speech[detect_non_silence(speech)] ** 2).mean()

        # 1. Convolve RIR
        if total_rir>0 and rir_apply_prob >= np.random.random():
            speech, _ = _convolve_rir(rir_scp, total_rir, speech, power)

        # 2. Add Noise
        if (total_noise>0 and noise_apply_prob >= np.random.random()):
            speech, _ = _add_noise(noise_scp,total_noise,speech, power,noise_db_low,noise_db_high)

        if len(speech.shape) > 1:  # multi-channel
            speech = speech[0]  # choose first channel
        ma = np.max(np.abs(speech))
        if ma > 1.0:
            speech /= ma

        sample['wav'] = torch.from_numpy(speech).unsqueeze(0).type_as(sample['wav'])

        yield sample

def compute_fbank(data,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          energy_floor=0.0,
                          sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def compute_mfcc(data,
                 num_mel_bins=23,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.0,
                 num_ceps=40,
                 high_freq=0.0,
                 low_freq=20.0):
    """ Extract mfcc

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.mfcc(waveform,
                         num_mel_bins=num_mel_bins,
                         frame_length=frame_length,
                         frame_shift=frame_shift,
                         dither=dither,
                         num_ceps=num_ceps,
                         high_freq=high_freq,
                         low_freq=low_freq,
                         sample_frequency=sample_rate)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def __tokenize_by_bpe_model(sp, txt):
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r'([\u4e00-\u9fff])')
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt) #chars = pattern.split(txt.upper())
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)

    return tokens


def tokenize(data,
             symbol_table,
             bpe_model=None,
             non_lang_syms=None,
             split_with_space=False):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, label, sample_rate}]
    """
    if non_lang_syms is not None:
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
    else:
        non_lang_syms = {}
        non_lang_syms_pattern = None

    if bpe_model is not None:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
    else:
        sp = None

    for sample in data:
        assert 'txt' in sample
        txt = sample['txt'].strip()
        if non_lang_syms_pattern is not None:
            parts = non_lang_syms_pattern.split(txt.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [txt]

        label = []
        tokens = []
        for part in parts:
            if part in non_lang_syms:
                tokens.append(part)
            else:
                if bpe_model is not None:
                    tokens.extend(__tokenize_by_bpe_model(sp, part))
                else:
                    if split_with_space:
                        part = part.split(" ")
                    for ch in part:
                        if ch == ' ':
                            ch = "▁"
                        tokens.append(ch)

        for ch in tokens:
            if ch in symbol_table:
                label.append(symbol_table[ch])
            elif '<unk>' in symbol_table:
                label.append(symbol_table['<unk>'])

        sample['tokens'] = tokens
        sample['label'] = label
        yield sample


def spec_aug(data, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10, max_w=80):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['feat'] = y
        yield sample


def spec_sub(data, max_t=20, num_t_sub=3):
    """ Do spec substitute
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of time substitute
            num_t_sub: number of time substitute to apply

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        for i in range(num_t_sub):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            # only substitute the earlier time chosen randomly for current time
            pos = random.randint(0, start)
            y[start:end, :] = x[start - pos:end - pos, :]
        sample['feat'] = y
        yield sample


def spec_trim(data, max_t=20):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]

        Args:
            data: Iterable[{key, feat, label}]
            max_t: max width of length trimming

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'feat' in sample
        x = sample['feat']
        assert isinstance(x, torch.Tensor)
        max_frames = x.size(0)
        length = random.randint(1, max_t)
        if length < max_frames / 2:
            y = x.clone().detach()[:max_frames - length]
            sample['feat'] = y
        yield sample


def shuffle(data, shuffle_size=10000):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, feat, label}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def sort(data, sort_size=500):
    """ Sort the data by feature length.
        Sort is used after shuffle and before batch, so we can group
        utts with similar lengths into a batch, and `sort_size` should
        be less than `shuffle_size`

        Args:
            data: Iterable[{key, feat, label}]
            sort_size: buffer size for sort

        Returns:
            Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= sort_size:
            buf.sort(key=lambda x: x['feat'].size(0))
            for x in buf:
                yield x
            buf = []
    # The sample left over
    buf.sort(key=lambda x: x['feat'].size(0))
    for x in buf:
        yield x


def static_batch(data, batch_size=16):
    """ Static batch the data by `batch_size`

        Args:
            data: Iterable[{key, feat, label}]
            batch_size: batch size

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000):
    """ Dynamic batch the data until the total frames in batch
        reach `max_frames_in_batch`

        Args:
            data: Iterable[{key, feat, label}]
            max_frames_in_batch: max_frames in one batch

        Returns:
            Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        assert 'feat' in sample
        assert isinstance(sample['feat'], torch.Tensor)
        new_sample_frames = sample['feat'].size(0)
        longest_frames = max(longest_frames, new_sample_frames)
        frames_after_padding = longest_frames * (len(buf) + 1)
        if frames_after_padding > max_frames_in_batch:
            yield buf
            buf = [sample]
            longest_frames = new_sample_frames
        else:
            buf.append(sample)
    if len(buf) > 0:
        yield buf


def batch(data, batch_type='static', batch_size=16, max_frames_in_batch=12000):
    """ Wrapper for static/dynamic batch
    """
    if batch_type == 'static':
        return static_batch(data, batch_size)
    elif batch_type == 'dynamic':
        return dynamic_batch(data, max_frames_in_batch)
    else:
        logging.fatal('Unsupported batch type {}'.format(batch_type))


def padding(data):
    """ Padding the data into training data

        Args:
            data: Iterable[List[{key, feat, label}]]

        Returns:
            Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        assert isinstance(sample, list)
        feats_length = torch.tensor([x['feat'].size(0) for x in sample],
                                    dtype=torch.int32)
        order = torch.argsort(feats_length, descending=True)
        feats_lengths = torch.tensor(
            [sample[i]['feat'].size(0) for i in order], dtype=torch.int32)
        sorted_feats = [sample[i]['feat'] for i in order]
        sorted_keys = [sample[i]['key'] for i in order]
        sorted_labels = [
            torch.tensor(sample[i]['label'], dtype=torch.int64) for i in order
        ]
        label_lengths = torch.tensor([x.size(0) for x in sorted_labels],
                                     dtype=torch.int32)

        padded_feats = pad_sequence(sorted_feats,
                                    batch_first=True,
                                    padding_value=0)
        padding_labels = pad_sequence(sorted_labels,
                                      batch_first=True,
                                      padding_value=-1)

        yield (sorted_keys, padded_feats, padding_labels, feats_lengths,
               label_lengths)
