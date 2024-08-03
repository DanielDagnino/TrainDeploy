#!/usr/bin/env python
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import timm
import torch
import torchaudio.compliance.kaldi as ta_kaldi
import torchvision.transforms as torch_vis_transf
from easydict import EasyDict
from torch import Tensor
from torch import nn
from torch.nn import Module

from BEATs import BEATs, BEATsConfig
from apis.utils.torch.dataparallel import rmv_module_dataparallel, is_model_dataparallel
from apis.utils.torch.decorators import preserve_trainability

# from EfficientAT.models.MobileNetV3 import get_model as get_mobilenet, get_ensemble_model
# from EfficientAT.helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
# from EfficientAT.models.preprocess import AugmentMelSTFT


class BaseModel(Module, ABC):
    @abstractmethod
    def forward(self, *x: np.ndarray) -> np.ndarray:
        """
        See forward of torch.nn.Module
        """
        raise NotImplementedError

    def n_parameters_grad(self) -> int:
        return sum(_.numel() for _ in self.parameters() if _.requires_grad)

    def n_parameters(self) -> int:
        return sum(_.numel() for _ in self.parameters())

    @property
    def data_type(self) -> int:
        return list(self.parameters())[0].dtype

    @abstractmethod
    def preprocessor(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    @preserve_trainability(stage="eval")
    def size_output(self, size: Tuple[int, ...], input_type_test: str) -> list:
        raise NotImplementedError

    @abstractmethod
    def load(self, load_model_fn: str):
        return None

    @property
    @abstractmethod
    def n_features(self) -> int:
        raise NotImplementedError


class AudioClf(BaseModel):
    def __init__(self, cfg, cfg_beats_pretr):
        super().__init__()
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)

        cfg = EasyDict(cfg)
        self.name = cfg.backbone.name
        self.n_classes = cfg.args.n_classes
        self.freeze_backbone = cfg.args.freeze_backbone
        self.sr = cfg.args.sr
        self.verbose = cfg.args.verbose

        # For stats information of input feats.
        self.mean_wf, self.std_wf = 0, 0
        self.mean_fb, self.std_fb = 0, 0
        self.n_stats = 0

        if self.name == "beats":
            assert self.sr == 16_000

            cfg = BEATsConfig(cfg=cfg_beats_pretr)
            BEATs_model = BEATs(cfg)
            self.backbone = BEATs_model

            self.predictor_dropout = None

            self.n_backbone_features = 768
            self.gem = None
            self.fc = nn.Linear(self.n_backbone_features, self.n_classes, bias=True)

        else:
            msg = f'Unknown backbone name = {self.name}'
            self.logger.error(msg)
            raise ValueError(msg)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if self.name == "ecaresnet50t":
            self.prep_trans = torch_vis_transf.Compose([
                torch_vis_transf.Normalize(
                    mean=self.backbone.default_cfg['mean'], std=self.backbone.default_cfg['std']),
            ])

    def backbone_forward(self, fbank):
        if self.name == "beats":
            fbank = fbank.unsqueeze(1)
            features = self.backbone.patch_embedding(fbank)
            features = features.reshape(features.shape[0], features.shape[1], -1)
            features = features.transpose(1, 2)
            features = self.backbone.layer_norm(features)
            features = self.backbone.post_extract_proj(features)
            x = self.backbone.dropout_input(features)
            x, layer_results = self.backbone.encoder(x, padding_mask=None)
            return x
        elif self.name == "mn40_as":
            return self.backbone.forward(fbank)[0]
        elif self.name == "ecaresnet50t":
            return self.backbone.forward(fbank)[-1]

    def preprocessor(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.name == "beats":
            fbank_mean = 17.276235528808066
            fbank_std = 4.7409234052119045
            fbanks = []
            # print("in - ", x.shape)
            for waveform in x:
                waveform = waveform.unsqueeze(0)
                # self.mean_wf += waveform.mean().item()
                # self.std_wf += waveform.std().item()
                waveform = (waveform - waveform.mean()) / (waveform.max() - waveform.mean())
                waveform = waveform * (2 ** 15)
                # print(waveform.dtype, waveform.min(), waveform.max(), waveform.mean(), waveform.std())
                # print(f"waveform.std() = {waveform.std()}")
                # print(f"waveform.shape = {waveform.shape}")

                fbank = ta_kaldi.fbank(
                    waveform, num_mel_bins=128, sample_frequency=16_000, frame_length=25, frame_shift=10)
                fbank = fbank.to(torch.float32)
                fbanks.append(fbank)
                # self.mean_fb += fbank.mean().item()
                # self.std_fb += fbank.std().item()
            # self.n_stats += len(x)
            # print("wf - ", self.mean_wf / self.n_stats, self.std_wf / self.n_stats, self.n_stats)
            # print("fb - ", self.mean_fb / self.n_stats, self.std_fb / self.n_stats, self.n_stats)
            fbank = torch.stack(fbanks, dim=0)
            fbank = (fbank - fbank_mean) / (2 * fbank_std)
            # print(f"fbank.std() = {fbank.std()}")
            # fbank.std() = 0.43130746483802795 hit_300k.mp3
            return fbank

        elif self.name == "mn40_as":
            old_shape = x.size()
            x = x.reshape(-1, old_shape[2])
            x = self.mel(x)
            x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
            return x

        elif self.name == "ecaresnet50t":
            fbank_mean = 17.276235528808066
            fbank_std = 4.7409234052119045
            fbanks = []
            for waveform in x:
                waveform = waveform.unsqueeze(0)
                waveform = (waveform - waveform.mean()) / (waveform.max() - waveform.mean())
                waveform = waveform * (2 ** 15)
                fbank = ta_kaldi.fbank(
                    waveform, num_mel_bins=128, sample_frequency=16_000, frame_length=25, frame_shift=10)
                fbank = torch.tensor(fbank).to(torch.float32)
                fbanks.append(fbank)
            fbank = torch.stack(fbanks, dim=0)
            fbank = (fbank - fbank_mean) / (2 * fbank_std)
            fbank = torch.stack([fbank, fbank, fbank], dim=1)
            return self.prep_trans(fbank)
        else:
            msg = f'Unknown backbone name = {self.name}'
            self.logger.error(msg)
            raise ValueError(msg)

    def print(self, msg):
        if self.verbose:
            print(msg)

    def load(self, load_model_fn: str):
        state_dict = torch.load(load_model_fn, map_location='cpu')['model']
        if not is_model_dataparallel(self):
            state_dict = rmv_module_dataparallel(state_dict)
        self.load_state_dict(state_dict, strict=True)

    def forward(self, x: np.ndarray) -> Tensor:
        self.print(f"input > {x.shape}")
        nb = x.shape[0]
        nd = self.n_backbone_features

        fbank = self.preprocessor(x)
        self.print(f"preprocessor > {fbank.shape}")

        out = self.backbone_forward(fbank)
        self.print(f"backbone_forward > {out.shape}")

        if self.gem:
            self.print(f"gem kernel_size > {self.gem.kernel_size}")
            out = self.gem(out).view(nb, -1, nd)
            self.print(f"gem > {out.shape}")

        # out = out.mean(1)
        # self.print(f"mean > {out.shape}")

        if self.predictor_dropout is not None:
            out = self.predictor_dropout(out)
        if self.fc:
            out = self.fc(out)
            self.print(f"predictor_dropout > {out.shape}")

        out = out.mean(1)
        self.print(f"mean > {out.shape}")
        self.print(f"fc > {out.shape}")

        return out

    @preserve_trainability(stage="eval")
    def size_output(self, size: Tuple[int, ...], input_type_test: str) -> list:
        dummy = torch.randn(size, dtype=eval(input_type_test))
        size_output = list(self.forward(dummy).shape)
        return size_output

    @property
    def n_features(self) -> int:
        return self.n_classes


if __name__ == "__main__":
    _n_classes = 1
    # Models from timm==0.8.2.dev0
    _test = [
        # (1, "mn40_as", 10 * 16_000),

        # (1, "ecaresnet50t", 10 * 16_000),

        (45, "beats", 5 * 16_000),
        # (5 * 45, "beats", 16_000),

        # (1, "beats", 10 * 16_000),
    ]
    for (_batch_size, _name, _nt) in _test:
        print(_name)

        _cfg = {
            "name": "AudioFP",
            "args": {
                "n_classes": 1,
                "sr": 16_000,
                "freeze_backbone": False,
                "verbose": True,

                # "img_size": (998, 128),
                # "gem_type": "avg",
                # "gem_train": False,
                # "gem_p": 3.,
            },
            "backbone": {
                "name": _name,
                "args": {
                    "features_only": True,
                    "pretrained": True,
                },
            },
            "resume": {
                "pretrained_model_fn": "/home/razor/MyModels/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
            },
        }

        # _cfg = {
        #     "name": "AudioFP",
        #     "args": {
        #         "n_classes": 1,
        #         "sr": 16_000,
        #         "freeze_backbone": False,
        #         "verbose": True,
        #     },
        # }

        model = AudioClf(_cfg)
        model.cuda()
        model.eval()
        print(10 * ' ', 'preprocessor        ', model.preprocessor)
        print(10 * ' ', 'n_features          ', model.n_features)
        print(10 * ' ', 'n_parameters        ', model.n_parameters())
        print(10 * ' ', 'n_parameters_grad   ', model.n_parameters_grad())

        optim_params = [{'params': model.parameters(), 'weight_decay': 1.e-3}]
        optimizer = torch.optim.SGD(optim_params, lr=0.1)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            _audio_input_16khz = torch.randn(_batch_size, _nt).to('cuda')
            output = model.forward(_audio_input_16khz)
            loss = torch.pow(output, 2).sum()

        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        assert output.ndim == 2
        assert output.shape[0] == _batch_size
        assert output.shape[1] == _n_classes
