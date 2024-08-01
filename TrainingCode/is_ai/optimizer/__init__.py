import inspect
import logging
from typing import Union

from easydict import EasyDict
from torch.nn import Module
from torch.optim import SGD, Adam, AdamW

RETURN_optimizer_builder = Union[SGD, Adam, AdamW]


def get_optimizer(model: Module, cfg: dict = None) -> RETURN_optimizer_builder:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    cfg = EasyDict(cfg)
    if not cfg:
        cfg.name = 'SGD'
        cfg.args = EasyDict(lr=0.1)
        logger.warning(f'Optimizer is not defined. Using default optimizer: {cfg.name} with lr: {cfg.args.lr}')

    if cfg.name == 'SGD':
        return SGD(model.parameters(), **cfg.args)
    elif cfg.name == 'Adam':
        return Adam(model.parameters(), **cfg.args)
    elif cfg.name == 'AdamW':
        return AdamW(model.parameters(), **cfg.args)
    else:
        msg = f'Not implemented optimizer: name = {cfg.name}'
        logger.error(msg)
        raise NotImplementedError(msg)
