import inspect
import logging
from typing import Union

from easydict import EasyDict
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR
from torch.optim.optimizer import Optimizer

RETURN_scheduler_builder = Union[LinearLR, StepLR, ReduceLROnPlateau]


def get_scheduler(optimizer: Optimizer, cfg: dict = None) -> RETURN_scheduler_builder:
    """get_scheduler"""
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    cfg = EasyDict(cfg)
    if not cfg:
        cfg.name = 'StepLR'
        step_size = 1_000_000
        cfg.args = EasyDict(step_size=step_size)
        logger.warning(f'Scheduler is not defined. Using default scheduler: {cfg.name}')
        logger.warning(f'step_size is not defined. Using default step_size: {step_size}')

    if cfg.name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, **cfg.args)
    elif cfg.name == 'StepLR':
        return StepLR(optimizer, **cfg.args)
    elif cfg.name == 'LinearLR':
        return LinearLR(optimizer, **cfg.args)
    else:
        msg = f'Not implemented scheduler: name = {cfg.name}'
        logger.error(msg)
        raise NotImplementedError(msg)


def set_optimizer_lr(optimizer: Optimizer, lr: float):
    """set_optimizer_lr"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
