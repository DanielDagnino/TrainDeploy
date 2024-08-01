#!/usr/bin/env python
import inspect
import logging
import math
import random

import torch
import torchaudio
import torchaudio.functional
from torch_pitch_shift import pitch_shift

from is_ai.is_ai_song.dataset.utils import check_mono_same_shape


# https://jonathanbgn.com/2021/08/30/audio-augmentation.html
# https://github.com/KentoNishi/torch-pitch-shift/blob/master/example.py
# https://github.com/asteroid-team/torch-audiomentations/blob/main/torch_audiomentations/core/transforms_interface.py


class RandomTimeShift:
    """Time shift augmentation"""

    def __init__(self, sample_rate, shift_sec, prob=0.5):
        self.shift_sec = shift_sec
        self.sample_rate = sample_rate
        self.prob = prob

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)

        if random.random() < self.prob:
            shift = int(random.random() * self.sample_rate * self.shift_sec)
            if shift > 0:
                audio1 = torch.roll(audio1, shifts=shift)
                if idxs1 is not None:
                    idxs1 = torch.roll(idxs1, shifts=shift)

        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        return idxs1, audio0, audio1


class RandomClip:
    def __init__(self, sample_rate, clip_length, apply_vad=False):
        self.clip_length = clip_length
        self.apply_vad = apply_vad
        self.vad = torchaudio.transforms.Vad(sample_rate=sample_rate, trigger_level=7.0)

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        audio_length = audio1.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio1 = audio1[offset:(offset + self.clip_length)]
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)

        if self.apply_vad:
            # FIXME
            raise NotImplementedError()
            # return idxs1, audio0, self.vad(audio1)
        else:
            return idxs1, audio0, audio1


class FlipAmplitude:
    """Globally multiply the signal by +-1"""

    def __init__(self, prob=0.5):
        self.prob = prob

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        if random.random() < self.prob:
            audio1 *= -1
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        return idxs1, audio0, audio1


class RandomSpeedChange:
    """Speed augmentation using SOX"""

    def __init__(self, sample_rate, pert_speed=0.1, prob=0.5):
        self.pert_speed = pert_speed
        self.sample_rate = sample_rate
        self.prob = prob

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        len_in_1 = len(audio1)
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)

        if random.random() < self.prob:
            speed_factor = 1 + (2 * random.random() - 1) * self.pert_speed
            sox_effects = [["speed", str(speed_factor)], ["rate", str(self.sample_rate)]]
            audio1 = audio1.reshape(1, len(audio1))
            audio1, _ = torchaudio.sox_effects.apply_effects_tensor(audio1, self.sample_rate, sox_effects)
            audio1 = audio1.reshape(-1)

        scale = (len_in_1 - 1.) / (len(audio1) - 1.)
        if idxs1 is not None:
            idxs1 = torch.tensor([idxs1[int(_ * scale)] for _ in range(len(audio1))]).long()

        if len(audio0) > len(audio1):
            audio0 = audio0[:len(audio1)]
        else:
            audio1 = audio1[:len(audio0)]

        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        return idxs1, audio0, audio1


class RandomPitchShift:
    def __init__(self, sample_rate, factor=12, prob=0.5):
        self.sample_rate = sample_rate
        self.factor = factor
        self.prob = prob

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        factor = (2 * random.random() - 1) * self.factor

        audio1 = audio1.reshape(1, 1, len(audio1))
        audio1 = pitch_shift(audio1, factor, self.sample_rate)
        audio1 = audio1.reshape(-1)

        check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        return idxs1, audio0, audio1


class RandomBackgroundNoise:
    def __init__(self, sample_rate, augment_files, min_snr_db=0, max_snr_db=15, prob=0.5, apply_both=False):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.apply_both = apply_both
        self.prob = prob

        self.augment_files = augment_files
        if len(self.augment_files) == 0:
            raise ValueError(f'No augmentation files provided.')

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, idxs1, audio0, audio1):
        if self.prob > random.random():
            check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)

            random_noise_file = random.choice(self.augment_files)
            effects = [
                ['remix', '1'],  # convert to mono
                ['rate', str(self.sample_rate)],  # resample
            ]
            noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
            noise = noise[0]
            audio_length = len(audio1)
            noise_length = len(noise)
            if noise_length > audio_length:
                offset = random.randint(0, noise_length - audio_length)
                noise = noise[offset:offset + audio_length]
            elif noise_length < audio_length:
                noise = torch.cat([noise, torch.zeros((audio_length - noise_length))], dim=-1)

            if self.apply_both:
                snr_db = random.randint(self.min_snr_db, self.max_snr_db)
                snr = math.exp(snr_db / 10)
                audio_power = audio0.norm(p=2)
                if audio_power == 0:
                    msg = "Found audio with full zero samples: audio_power = 0."
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                else:
                    noise_power = noise.norm(p=2)
                    scale = snr * noise_power / audio_power
                    audio0 = (scale * audio0 + noise) / 2

            snr_db = random.randint(self.min_snr_db, self.max_snr_db)
            snr = math.exp(snr_db / 10)
            audio_power = audio1.norm(p=2)
            if audio_power == 0:
                msg = "Found audio with full zero samples: audio_power = 0."
                self.logger.error(msg)
                raise RuntimeError(msg)
            else:
                noise_power = noise.norm(p=2)
                scale = snr * noise_power / audio_power
                audio1 = (scale * audio1 + noise) / 2

            check_mono_same_shape(self.__class__.__qualname__, audio0, audio1)
        return idxs1, audio0, audio1
