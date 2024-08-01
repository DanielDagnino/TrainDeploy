import math
import os

import torch
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict

from is_ai.is_ai_voice.dataset.ds_infer_clf import DatasetAudioMixerInfer
from is_ai.is_ai_voice.model.base_clf import AudioClf
from is_ai.model.helpers import load_checkpoint
from is_ai.utils.general.custom_yaml import init_custom_yaml
from is_ai.utils.general.modifier import dict_modifier


class Initiate:
    def __init__(self, args):
        print('Configuration')
        init_custom_yaml()
        self.cfg = yaml.load(open(args.cfg_fn), Loader=yaml.Loader)
        self.cfg = dict_modifier(config=self.cfg, modifiers="modifiers",
                                 pre_modifiers={"HOME": os.path.expanduser("~")})
        self.cfg = EasyDict(self.cfg)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("Build model")
        if self.cfg.engine.model.name == AudioClf.__name__:
            print(f"{AudioClf.__name__} building")
            self.model = AudioClf(self.cfg.engine.model)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            msg = f'Unknown model name = {self.cfg.name}'
            print(msg)
            raise ValueError(msg)

        print("Load model")
        _, _, _, _ = load_checkpoint(
            self.model, self.cfg.engine.model.resume.load_model_fn, None, None, None, torch.device(device), False,
            False, True)
        self.model.eval()

        print("Dataset")
        if self.cfg.dataset.name == DatasetAudioMixerInfer.__name__:
            self.dataset = DatasetAudioMixerInfer(**self.cfg.dataset.get("test"))
        else:
            raise ValueError(f"Not implemented dataset {self.cfg.dataset.name}")

        print('Compute results')
        if self.cfg.cnn_benchmark:
            cudnn.benchmark = True

    def split_array(self, x, n_seg, n_hop):
        segments = []
        idx = 0
        while idx + n_seg <= len(x):
            segments.append(x[idx:idx + n_seg])
            idx += n_hop
        return segments

    def reducer(self, x, length, hop, func="max"):
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
        _pred = self.reducer(_pred, length=5, hop=5, func="max")
        _pred = torch.sigmoid(smooth * _pred).mean().item()
        _pred = _pred if not math.isnan(_pred) else 2
        return _pred
