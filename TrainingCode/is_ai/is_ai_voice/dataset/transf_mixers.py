#!/usr/bin/env python
import inspect
import logging
import math
import random
from typing import Tuple, List, Dict

import numpy as np
import torch
from torch import Tensor

from is_ai.is_ai_song.dataset.utils import MusicBank, MusicSplits, Mix
from is_ai.is_ai_song.dataset.utils import check_mono, check_same_shape
from data_collection.utils.audio.torch_manipulate import read_to_waveform


# FIXME: hop_length is not used.


def db_to_prod(db):
    return math.pow(10., db / 20.)


class TestMixer:
    def __init__(self,
                 sample_rate: int,
                 audio_length_sec: float,
                 ):
        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec

    def __call__(self, audio_bank: MusicBank) -> Tuple[Tensor, Mix]:
        return torch.randn(int(self.audio_length_sec * self.sample_rate)), Mix.VocalOrigInstrModMel


class GeneratedVocalMixGeneratedInstruments:
    def __init__(self,
                 sample_rate: int,
                 audio_length_sec: float,
                 factor_mix_db: Tuple[float, float] = (-5., 5.),
                 ):
        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec
        self.factor_mix_db = factor_mix_db

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, audio_bank: MusicBank) -> Tuple[Tensor, Mix]:
        _, tmp = read_to_waveform(
            audio_bank.vocal_gen, mono=True, sr=self.sample_rate,
            duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
        factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[0]
        audio_sum = db_to_prod(factor_mix_db) * tmp

        _, tmp = read_to_waveform(
            audio_bank.instr_gen_wmel, mono=True, sr=self.sample_rate,
            duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
        factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[0]
        audio_sum += db_to_prod(factor_mix_db) * tmp

        check_mono(self.__class__.__qualname__, audio_sum)
        return audio_sum, Mix.VocalModInstrModMel


class GeneratedVocalMixOriginalInstruments:
    def __init__(self,
                 sample_rate: int,
                 audio_length_sec: float,
                 factor_mix_db: Tuple[float, float] = (-5., 5.),
                 prob_keep_inst: Dict[MusicSplits, float] = {
                     MusicSplits.bass: 0.90,
                     MusicSplits.drums: 0.90,
                     MusicSplits.other: 0.90
                 }
                 ):
        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec
        self.factor_mix_db = factor_mix_db
        self.prob_keep_inst = prob_keep_inst

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, audio_bank: MusicBank) -> Tuple[Tensor, Mix]:
        if len(self.prob_keep_inst) != 3:
            msg = f'Length of prob_keep_inst must be 3. However: \n' \
                  f'len(prob_keep_inst) = {len(self.prob_keep_inst)} \n'
            self.logger.error(msg)
            raise ValueError(msg)

        _, tmp = read_to_waveform(
            audio_bank.vocal_gen, mono=True, sr=self.sample_rate,
            duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
        factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[0]
        audio_sum = db_to_prod(factor_mix_db) * tmp

        for audio_split, audio_fn in audio_bank.get_instruments_orig().items():
            if random.random() < self.prob_keep_inst[audio_split]:
                factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[
                    0]
                _, tmp = read_to_waveform(audio_fn, mono=True, sr=self.sample_rate,
                                          duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
                audio_sum += db_to_prod(factor_mix_db) * tmp

        check_mono(self.__class__.__qualname__, audio_sum)
        return audio_sum, Mix.VocalModInstrOrig


class OriginalVocalMixGeneratedInstruments:
    def __init__(self,
                 sample_rate: int,
                 audio_length_sec: float,
                 prob_with_melody: float,
                 random_gen_instrs: List[str],
                 factor_mix_db: Tuple[float, float] = (-5., 5.),
                 ):
        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec
        self.prob_with_melody = prob_with_melody
        self.factor_mix_db = factor_mix_db
        self.random_gen_instrs = random_gen_instrs

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, audio_bank: MusicBank) -> Tuple[Tensor, Mix]:
        _, tmp = read_to_waveform(
            audio_bank.vocals, mono=True, sr=self.sample_rate,
            duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
        factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[0]
        audio_sum = db_to_prod(factor_mix_db) * tmp

        factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[0]
        if random.random() < self.prob_with_melody:
            _, tmp = read_to_waveform(
                audio_bank.instr_gen_wmel, mono=True, sr=self.sample_rate,
                duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
            audio_sum += db_to_prod(factor_mix_db) * tmp

            check_mono(self.__class__.__qualname__, audio_sum)
            return audio_sum, Mix.VocalOrigInstrModMel
        else:
            fn = np.random.choice(self.random_gen_instrs)
            _, tmp = read_to_waveform(fn, mono=True, sr=self.sample_rate,
                                      duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
            audio_sum += db_to_prod(factor_mix_db) * tmp

            check_mono(self.__class__.__qualname__, audio_sum)
            return audio_sum, Mix.VocalOrigInstrModNoMel


class OriginalVocalMixOriginalInstruments:
    def __init__(self,
                 sample_rate: int,
                 audio_length_sec: float,
                 factor_mix_db: Tuple[float, float] = (-5., 5.),
                 prob_keep_inst: Tuple[MusicSplits, float] = {
                     MusicSplits.bass: 0.90, MusicSplits.drums: 0.90, MusicSplits.other: 0.90
                 }
                 ):
        self.sample_rate = sample_rate
        self.audio_length_sec = audio_length_sec
        self.factor_mix_db = factor_mix_db
        self.prob_keep_inst = prob_keep_inst

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def __call__(self, audio_bank: MusicBank) -> Tuple[Tensor, Mix]:
        if len(self.prob_keep_inst) != 3:
            msg = f'Length of audio_bank must be equal to the length of prob_keep_inst. However: \n' \
                  f'len(prob_keep_inst) = {len(self.prob_keep_inst)} \n'
            self.logger.error(msg)
            raise ValueError(msg)

        _, tmp = read_to_waveform(
            audio_bank.vocals, mono=True, sr=self.sample_rate,
            duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
        factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[0]
        audio_sum = db_to_prod(factor_mix_db) * tmp

        for audio_split, audio_fn in audio_bank.get_instruments_orig().items():
            if random.random() < self.prob_keep_inst[audio_split]:
                _, tmp = read_to_waveform(audio_fn, mono=True, sr=self.sample_rate,
                                          duration_min=self.audio_length_sec, duration_max=10 * self.audio_length_sec)
                factor_mix_db = random.random() * (self.factor_mix_db[1] - self.factor_mix_db[0]) + self.factor_mix_db[
                    0]
                audio_sum += db_to_prod(factor_mix_db) * tmp

        check_mono(self.__class__.__qualname__, audio_sum)
        return audio_sum, Mix.VocalOrigInstrOrig


class SplitSegment:
    def __init__(self, segment_length, hop_length, n_segments_per_sample):
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.n_segments_per_sample = n_segments_per_sample

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

    def split(self, audio, idxs):
        length_fix = self.segment_length * (len(audio) // self.segment_length)
        audio = audio[:length_fix].split(self.segment_length, dim=0)
        audio = torch.stack(audio, dim=0)
        if idxs is not None:
            idxs = idxs[:length_fix].split(self.segment_length, dim=0)
            idxs = torch.stack(idxs, dim=0)
            return idxs, audio
        else:
            return None, audio

    def __call__(self, idxs0, idxs1, audio0, audio1):
        check_mono(self.__class__.__qualname__, audio0)
        check_mono(self.__class__.__qualname__, audio1)

        length0, length1 = len(audio0), len(audio1)
        if not (length0 >= self.segment_length and length1 >= self.segment_length):
            msg = f'Minimum length must be larger than segment_length. However: \n' \
                  f'segment_length = {self.segment_length} \n' \
                  f'length0 = {length0}, length1 = {length1} \n'
            self.logger.error(msg)
            raise ValueError(msg)

        # split and stack
        idxs0, audio0 = self.split(audio0, idxs0)
        idxs1, audio1 = self.split(audio1, idxs1)

        # select segments randomly
        idx_select = np.random.choice(audio0.shape[0], size=self.n_segments_per_sample, replace=False)
        audio0 = audio0[idx_select]
        if idxs0 is not None:
            idxs0 = idxs0[idx_select]
        audio1 = audio1[idx_select]
        if idxs1 is not None:
            idxs1 = idxs1[idx_select]

        if audio0.ndim != 2:
            msg = f'Audio must have two dimensions. However, audio0.size = {audio0.shape} \n'
            self.logger.error(msg)
            raise ValueError(msg)

        check_same_shape(self.__class__.__qualname__, audio0, audio1)
        return idxs0, idxs1, audio0, audio1

    # def call_inference(self, audio0):
    #     check_mono(self.__class__.__qualname__, audio0)
    #
    #     length0 = len(audio0)
    #     if not (length0 >= self.segment_length):
    #         msg = f'Minimum length must be larger than segment_length. However: \n' \
    #               f'segment_length = {self.segment_length} \n'
    #         self.logger.error(msg)
    #         raise ValueError(msg)
    #
    #     # split and stack
    #     _, audio0 = self.split(audio0, None)
    #
    #     # select segments randomly
    #     idx_select = np.random.choice(audio0.shape[0], size=self.n_segments_per_sample, replace=False)
    #     audio0 = audio0[idx_select]
    #
    #     if audio0.ndim != 2:
    #         msg = f'Audio must have two dimensions. However, audio0.size = {audio0.shape} \n'
    #         self.logger.error(msg)
    #         raise ValueError(msg)
    #
    #     check_mono(self.__class__.__qualname__, audio0)
    #     return audio0
