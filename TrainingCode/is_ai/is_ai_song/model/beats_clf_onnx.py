#!/usr/bin/env python
import inspect
import logging

import numpy as np
import torch
from easydict import EasyDict
from torch import Tensor
from torch import nn
from torch.nn import Module

from BEATs import BEATs, BEATsConfig
from is_ai.model.np_kaldi import ta_kaldi_fbank
from is_ai.utils.torch.dataparallel import rmv_module_dataparallel, is_model_dataparallel


class AudioClf(Module):
    def __init__(self, cfg):
        super().__init__()
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        cfg = EasyDict(cfg)
        self.name = cfg.backbone.name
        self.n_classes = cfg.args.n_classes
        self.sr = cfg.args.sr

        if self.name == "beats":
            assert self.sr == 16_000

            checkpoint = torch.load(cfg.resume.pretrained_model_fn)
            checkpoint['cfg']["finetuned_model"] = False
            checkpoint['cfg']["dropout_input"] = 0.10
            del checkpoint['cfg']["predictor_dropout"]
            predictor_dropout = 0.10
            del checkpoint['cfg']["predictor_class"]
            checkpoint['cfg']["dropout"] = 0.10

            cfg = BEATsConfig(checkpoint['cfg'])
            BEATs_model = BEATs(cfg)
            BEATs_model.load_state_dict(checkpoint['model'], strict=False)
            self.backbone = BEATs_model
            self.predictor_dropout = nn.Dropout(predictor_dropout)

            self.n_backbone_features = 768
            self.gem = None
            self.fc = nn.Linear(self.n_backbone_features, self.n_classes, bias=True)

        else:
            msg = f'Unknown backbone name = {self.name}'
            self.logger.error(msg)
            raise ValueError(msg)

    def backbone_forward(self, fbank):
        fbank = fbank.unsqueeze(1)
        features = self.backbone.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.backbone.layer_norm(features)
        features = self.backbone.post_extract_proj(features)
        x = self.backbone.dropout_input(features)
        x, layer_results = self.backbone.encoder(x, padding_mask=None)
        return x

    def preprocessor(self, x: np.ndarray) -> np.ndarray:
        fbank_mean = 17.276235528808066
        fbank_std = 4.7409234052119045
        fbanks = []
        for waveform in x:
            # waveform = waveform.unsqueeze(0)
            waveform = waveform.reshape(1, waveform.shape[0])
            waveform = (waveform - waveform.mean()) / (waveform.max() - waveform.mean())
            waveform = waveform * (2 ** 15)

            fbank = ta_kaldi_fbank(waveform, num_mel_bins=128, sample_frequency=16_000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = np.stack(fbanks, axis=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def load(self, load_model_fn: str):
        state_dict = torch.load(load_model_fn, map_location='cpu')['model']
        if not is_model_dataparallel(self):
            state_dict = rmv_module_dataparallel(state_dict)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, fbank: np.ndarray) -> Tensor:
        # fbank = self.preprocessor(x)
        fbank = torch.tensor(fbank, dtype=torch.float32)
        out = self.backbone_forward(fbank)
        out = self.fc(out)
        out = out.mean(1)
        return out
