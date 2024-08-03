# !/usr/bin/env python
import inspect
import json
import logging
import math
import os
from io import BytesIO
from typing import Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict
from torch import Tensor
from torch.utils.data import Dataset

from apis.utils.general.custom_yaml import init_custom_yaml
from apis.utils.general.modifier import dict_modifier
from apis.utils.torch.dataparallel import rmv_module_dataparallel, is_model_dataparallel
from apis.utils.audio.torch_manipulate import read_to_waveform
from apis.clf_ai.model.base_clf import AudioClf


class DatasetAudioMixerInfer(Dataset):
    def __init__(self,
                 io_file: BytesIO,
                 sample_rate: int,
                 segment_length_sec: float,
                 segment_length_max_sec: Optional[float],
                 hop_length_sec: float,
                 verbose: bool = False):

        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.verbose = verbose

        self.sample_rate = sample_rate
        self.segment_length_sec = segment_length_sec
        self.segment_length_max_sec = segment_length_max_sec
        self.hop_length_sec = hop_length_sec
        self.segment_length = int(sample_rate * segment_length_sec)
        self.hop_length = int(sample_rate * hop_length_sec)

        self.io_file = io_file

    def __len__(self) -> int:
        return 1

    def chunker(self, audio):
        length_fix = self.segment_length * (len(audio) // self.segment_length)
        audio = audio[:length_fix].split(self.segment_length, dim=0)
        # FIXME: add tail
        audio = torch.stack(audio, dim=0)
        return audio

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Tensor, float]:
        try:
            sr, audio = read_to_waveform(
                self.io_file, mono=True, sr=self.sample_rate, dtype=torch.float32, trim=True,
                duration_min=self.segment_length_sec, duration_max=self.segment_length_max_sec)
            if audio is None:
                msg = f'It was not possible to load any data.'
                self.logger.error(msg)
                raise RuntimeError(msg)
            duration = len(audio) / sr
            audio = self.chunker(audio)
            return audio, duration
        except Exception as excpt:
            self.logger.error(f"Some error with reading io_file")
            self.logger.error(excpt)


class Initiate:
    def __init__(self, io_file: Optional[BytesIO], args: EasyDict, cfg_beats_pretr_fn: str):
        self.logger = logging.getLogger(
            __name__ + ': ' + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        self.logger.info('Configuration')
        init_custom_yaml()
        self.cfg = yaml.load(open(args.cfg_fn), Loader=yaml.Loader)
        self.cfg = dict_modifier(config=self.cfg, modifiers="modifiers",
                                 pre_modifiers={"HOME": os.path.expanduser("~")})
        self.cfg = EasyDict(self.cfg)

        cfg_beats_pretr = json.load(open(cfg_beats_pretr_fn))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.logger.info("Build model")
        if self.cfg.engine.model.name == AudioClf.__name__:
            self.logger.info(f"{AudioClf.__name__} building")
            self.model = AudioClf(self.cfg.engine.model, cfg_beats_pretr)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            msg = f'Unknown model name = {self.cfg.name}'
            self.logger.info(msg)
            raise ValueError(msg)

        self.logger.info("Load model")
        checkpoint = torch.load(self.cfg.engine.model.resume.load_model_fn, map_location=device)
        state_dict = checkpoint['state_dict']
        if not is_model_dataparallel(self.model):
            state_dict = rmv_module_dataparallel(state_dict)
            state_dict = rmv_module_dataparallel(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.logger.info("Dataset")
        if self.cfg.dataset.name == DatasetAudioMixerInfer.__name__:
            self.dataset = DatasetAudioMixerInfer(io_file=io_file, **self.cfg.dataset.get("test"))
        else:
            raise ValueError(f"Not implemented dataset {self.cfg.dataset.name}")

        self.logger.info('Compute results')
        if self.cfg.cnn_benchmark:
            cudnn.benchmark = True

    def split_array(self, x, n_seg, n_hop):
        segments = []
        idx = 0
        while idx + n_seg <= len(x):
            segments.append(x[idx:idx + n_seg])
            idx += n_hop
        return segments

    def reducer(self, x, length=3, hop=1, func="max"):
        if len(x) > length:
            x = self.split_array(x, length, hop)
            x = torch.stack(x, dim=0)

        if func == "max":
            x = torch.tensor([_.max() for _ in x])
        else:
            x = torch.tensor([_.mean() for _ in x])
        return x

    def pred_reducer(self, _pred, smooth=1):
        _pred = torch.cat(_pred, dim=0)

        # _pred = reducer(_pred, func="mean")
        _pred = torch.sigmoid(smooth * _pred)
        # _pred = reducer(_pred, func="max")

        _pred = _pred.mean()
        _pred = _pred.item()
        _pred = _pred if not math.isnan(_pred) else 2
        return _pred
