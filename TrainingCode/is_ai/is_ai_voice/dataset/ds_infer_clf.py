#!/usr/bin/env python
import inspect
import json
import logging
from typing import List, Tuple, Optional

import torch
from path import Path
from torch import Tensor
from torch.utils.data import Dataset

from data_collection.utils.audio.torch_manipulate import read_to_waveform
from is_ai.utils.logger.logger import setup_logging


class DatasetAudioMixerInfer(Dataset):
    def __init__(self,
                 base_dir: str,
                 lbl_to_paths: List[Tuple[str, str]],
                 sample_rate: int,
                 segment_length_sec: float,
                 segment_length_max_sec: Optional[float],
                 hop_length_sec: float,
                 verbose: bool = False):

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.base_dir = Path(base_dir)
        self.verbose = verbose

        self.sample_rate = sample_rate
        self.segment_length_sec = segment_length_sec
        self.segment_length_max_sec = segment_length_max_sec
        self.hop_length_sec = hop_length_sec
        self.segment_length = int(sample_rate * segment_length_sec)
        self.hop_length = int(sample_rate * hop_length_sec)

        # Get files.
        self.samples = []
        self.lbls = []
        for lbl, paths in lbl_to_paths:
            tmp = json.load(open(paths))
            self.samples += tmp
            self.lbls.extend(len(tmp) * [lbl])
        self.logger.info(f'len(self.samples) = {len(self.samples)}')

    def __len__(self):
        return len(self.samples)

    def chunker(self, audio):
        length_fix = self.segment_length * (len(audio) // self.segment_length)
        audio = audio[:length_fix].split(self.segment_length, dim=0)
        audio = torch.stack(audio, dim=0)
        return audio

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, str, Tensor]:
        while self.samples:
            path, lbl = self.samples.pop(), self.lbls.pop()
            try:
                sr, audio = read_to_waveform(
                    path, mono=True, sr=self.sample_rate, dtype=torch.float32, trim=True,
                    duration_min=self.segment_length_sec, duration_max=self.segment_length_max_sec)
                if audio is None:
                    msg = f'It was not possible to load any data.'
                    self.logger.error(msg)
                    raise RuntimeError(msg)
                audio = self.chunker(audio)
                return Path(path).relpath(self.base_dir), lbl, audio
            except Exception as excpt:
                print(f"Some error with path = {path}")
                print(excpt)
        else:
            raise StopIteration


if __name__ == '__main__':
    setup_logging()

    ds = DatasetAudioMixerInfer(
        base_dir='/home/razor/MyData',
        lbl_to_paths=[
            ("ai", '/home/razor/MyTmp/DeepFakeDetector/splits/clf/clf/split_2/ai_song_fns_test.json'),
            ("non_ai", '/home/razor/MyTmp/DeepFakeDetector/splits/clf/clf/split_2/non_ai_song_fns_test.json')
        ],
        sample_rate=16_000,
        segment_length_sec=10,
        segment_length_max_sec=None,
        hop_length_sec=2,
        verbose=True
    )
    print(len(ds))
    for _idx, (_artist, _title, _idd, _text_search, _audio) in enumerate(ds):
        print(_idx, _artist, _idd, _audio.shape)
