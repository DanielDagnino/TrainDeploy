#!/usr/bin/env python
"""
This code has been done using the following repos:
    PyTorch implementation: https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
    Original NFNets in JAX: https://github.com/deepmind/deepmind-research/blob/master/nfnets/optim.py
"""
import inspect
import logging

import torch


def unitwise_norm(x, norm_type=2.0):
    if x.ndim <= 1:
        return x.norm(norm_type)
    else:
        # works for nn.ConvNd and nn,Linear where output dim is first in the kernel/weight tensor
        # might need special cases for other weights (possibly MHA) where this may not be true
        return x.norm(norm_type, dim=tuple(range(1, x.ndim)), keepdim=True)


class Clipper:
    def __init__(self, name=None, args=None, rank=0):
        self.logger = logging.getLogger(
            __name__ + ": " + self.__class__.__qualname__ + '-' + inspect.currentframe().f_code.co_name)
        self.name = name
        self.args = args
        self.rank = rank
        self.types = ['TorchClip', 'AGC']
        if self.name not in [None, "None", *self.types]:
            msg = f'Wrong clipper name {name}.'
            self.logger.error(msg)
            raise NotImplementedError(msg)

        if self.name == 'TorchClip':
            if not self.args.max_norm:
                raise ValueError()
        elif self.name == 'AGC':
            if not self.args.norm_type:
                raise ValueError()
            if not self.args.clip_factor:
                raise ValueError()
            if not self.args.eps:
                raise ValueError()

        self.logger.info(f"Created clipper: {name}")

    def is_not_null(self):
        if self.name in self.types:
            return True
        return False

    def apply_to_grad(self, model):
        if self.name == 'TorchClip':
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_norm, error_if_nonfinite=False)
            if norm.isnan():
                self.logger.warning(f"Gradient norm is NaN. norm={norm}")
            if norm > self.args.max_norm:
                self.logger.info(f"clip_grad_norm_: norm={norm} to max_norm={self.args.max_norm}")
        elif self.name == 'AGC':
            parameters = model.parameters()
            if isinstance(parameters, torch.Tensor):
                parameters = [parameters]
            for p in parameters:
                if p.grad is None:
                    continue
                p_data = p.detach()
                g_data = p.grad.detach()
                max_norm = unitwise_norm(p_data, norm_type=self.args.norm_type).clamp_(min=self.args.eps).mul_(
                    self.args.clip_factor)
                if torch.isnan(max_norm):
                    msg = f"NaN found while clipping"
                    self.logger.error(msg)
                    raise ValueError(msg)
                grad_norm = unitwise_norm(g_data, norm_type=self.args.norm_type)
                clipped_grad = g_data * (max_norm / grad_norm.clamp(min=1e-6))
                new_grads = torch.where(torch.gt(max_norm, grad_norm), g_data, clipped_grad)
                p.grad.detach().copy_(new_grads)


# if __name__ == "__main__":
#     from torch.optim import SGD
#     from is_ai_song.model_dscptr import ModelBuilder
#     from is_ai_song.utils.helpers import mock_image
#
#     _cfg = {
#         "name": "ModelDescriptor",
#         "args": {"add_bn": False, "gem_p": 1, "embedding_size": 256},
#         "backbone": {
#             "name": "eca_nfnet_l2",
#             "args": {"pretrained": False, "features_only": True},
#         },
#     }
#     _model = ModelBuilder(_cfg)
#
#     _optim = SGD(_model.parameters(), 1e-3)
#     _optim = Clipper(_model.parameters(), _optim, ignore_agc=("fc",))
#     _optim.zero_grad()
#     _model(mock_image(batch_size=2)).sum().backward()
#     _optim.step()
