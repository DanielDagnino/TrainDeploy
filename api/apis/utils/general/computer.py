import inspect
import logging
import os

import torch


def get_max_number_workers(number_workers: int = 0) -> int:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    if number_workers <= 0:
        return os.cpu_count()
    elif number_workers > os.cpu_count():
        logger.warning('number_workers=%s bigger than number of available workers %s. '
                       'The numbers of workers is set to the number available workers.', str(number_workers),
                       str(os.cpu_count()))
        return os.cpu_count()
    else:
        logger.warning('Number of workers is lower that the available ones:')
        logger.warning('\tNumber available workers = %s', str(os.cpu_count()))
        logger.warning('\tNumber workers = %s', str(number_workers))
        return number_workers


def choose_device(device: str = None) -> torch.device:
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        if device == 'cpu':
            logger.warning("CUDA is not available.")
    elif device[:4].lower() == 'cuda':
        if not torch.cuda.is_available():
            logging.error("CUDA is not available.")
            raise ValueError(__name__ + ": " + choose_device.__name__)
    elif device.lower() == 'cpu':
        return torch.device(device)
    else:
        logger.error('Not valid device value %s. Device must be cpu or cuda or cuda: <int>.', str(device))
        raise ValueError(__name__ + ": " + choose_device.__name__)

    if torch.cuda.is_available() and (device.lower() == 'cpu'):
        logger.warning('GPU available but using the CPU')

    if device == 'cpu':
        logger.warning("CUDA is not being used")
    else:
        logger.info("CUDA is being used")

    return torch.device(device)
