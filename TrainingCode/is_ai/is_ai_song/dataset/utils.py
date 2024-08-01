from enum import Enum, unique
from typing import Dict
import logging
import torch
from path import Path
from torch import Tensor


@unique
class Mix(Enum):
    VocalOrigInstrOrig = 0
    VocalOrigInstrModMel = 1
    VocalOrigInstrModNoMel = 2
    VocalModInstrOrig = 3
    VocalModInstrModMel = 4
    VocalModInstrModNoMel = 5


@unique
class MusicSplits(Enum):
    vocals = 0
    bass = 1
    drums = 2
    other = 3


class MusicBank:
    def __init__(self,
                 orig: Path, vocals: Path, bass: Path, drums: Path, other: Path, vocal_gen: Path, instr_gen_wmel: Path):
        self.orig = orig
        self.vocals = vocals
        self.bass = bass
        self.drums = drums
        self.other = other
        self.vocal_gen = vocal_gen
        self.instr_gen_wmel = instr_gen_wmel

    def get(self, instr):
        return self.__class__.__dict__[instr]

    def get_instruments_orig(self) -> Dict[MusicSplits, Path]:
        return {MusicSplits.bass: self.bass, MusicSplits.drums: self.drums, MusicSplits.other: self.other}

    def get_split_orig(self) -> Dict[MusicSplits, Path]:
        return {MusicSplits.vocals: self.vocals, MusicSplits.bass: self.bass, MusicSplits.drums: self.drums,
                MusicSplits.other: self.other}


def torch_join_segments(img: torch.Tensor, ib0: int = 0, ib1: int = 1):
    img = img[ib0:ib1]
    img = img.reshape(1, -1)
    return img


def check_mono(name, audio: Tensor):
    if audio.ndim != 1:
        msg = f'{name} - Audio must be mono-channel. However, audio0.size = {audio.shape} \n'
        logging.error(msg)
        raise ValueError(msg)


def check_mono_same_shape(name, audio0: Tensor, audio1: Tensor):
    check_same_shape(name, audio0, audio1)
    check_mono(name, audio0)


def check_same_shape(name, audio0: Tensor, audio1: Tensor):
    if audio0.shape != audio1.shape:
        msg = f'{name} - Both audio data must have the same dim and lengths. However: \n' \
              f'audio0.shape = {audio0.shape} \n' \
              f'audio1.shape = {audio1.shape} \n'
        logging.error(msg)
        raise ValueError(msg)
