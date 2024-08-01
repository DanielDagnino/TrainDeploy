#!/usr/bin/env python
import inspect
import logging
import random
import tempfile

import numpy as np
import pydub
import scipy.io.wavfile
import soundfile as sf
import torch

from is_ai.is_ai_song.dataset.utils import check_mono_same_shape


class AudioEncoding:
    """
    Change the audio codec that can modify the original amplitude of the waveform.
    Note: It is better to add this augmentation the first, or ensure the audio is not clipped.
    """

    def __init__(self, sample_rate, prob, codec='mp3', bits=16):
        self.codec = codec.lower()
        self.bits = bits
        if self.codec not in ['linear16', 'mp3']:
            raise ValueError(f'Not valid codec. Valid codecs: LINEAR16 and MP3')
        if self.bits != 16:
            raise ValueError(f'Not valid bits. Valid bits: 16')
        self.prob = prob
        self.sample_rate = sample_rate

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        if random.random() < self.prob:
            check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)

            x = audio0
            length = x.shape[-1]
            tmp_file = tempfile.TemporaryFile()
            dtype = x.dtype
            device = x.device
            x = x.cpu().numpy()
            x_max = np.max(np.abs(x)) + np.finfo(x.dtype).eps
            x /= x_max
            x = (x * (2 ** (self.bits - 1) - 1)).astype(np.int16)
            scipy.io.wavfile.write(tmp_file, self.sample_rate, x)
            x = pydub.AudioSegment.from_file(tmp_file)
            # bitrate (64, 92, 128, 256, 312k...)
            bitrate = random.choice(['16k', '32k', '64k', '128k'])
            if self.codec == 'mp3':
                x.export(tmp_file, format='mp3', bitrate=bitrate)
                x = pydub.AudioSegment.from_mp3(tmp_file)
                x.export(tmp_file, format="wav", bitrate=bitrate)
                x, _ = sf.read(tmp_file)
            else:
                x.export(tmp_file, format='wav', bitrate=bitrate)
                x, _ = sf.read(tmp_file)
            x = x[:length]
            x = x_max * (x / (np.max(np.abs(x)) + np.finfo(x.dtype).eps))
            x = torch.tensor(x, device=device, dtype=dtype)
            audio0 = x

            x = audio1
            length = x.shape[-1]
            tmp_file = tempfile.TemporaryFile()
            dtype = x.dtype
            device = x.device
            x = x.cpu().numpy()
            x_max = np.max(np.abs(x)) + np.finfo(x.dtype).eps
            x /= x_max
            x = (x * (2 ** (self.bits - 1) - 1)).astype(np.int16)
            scipy.io.wavfile.write(tmp_file, self.sample_rate, x)
            x = pydub.AudioSegment.from_file(tmp_file)
            # bitrate (64, 92, 128, 256, 312k...)
            bitrate = random.choice(['16k', '32k', '64k', '128k'])
            if self.codec == 'mp3':
                x.export(tmp_file, format='mp3', bitrate=bitrate)
                x = pydub.AudioSegment.from_mp3(tmp_file)
                x.export(tmp_file, format="wav", bitrate=bitrate)
                x, _ = sf.read(tmp_file)
            else:
                x.export(tmp_file, format='wav', bitrate=bitrate)
                x, _ = sf.read(tmp_file)
            x = x[:length]
            x = x_max * (x / (np.max(np.abs(x)) + np.finfo(x.dtype).eps))
            x = torch.tensor(x, device=device, dtype=dtype)
            audio1 = x

            check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)

        return idxs1, audio0, audio1
