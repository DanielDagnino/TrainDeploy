#!/usr/bin/env python
"""
CLI to train and evaluate an Image Descriptors model
"""
import argparse
import datetime
import getpass
import json
import logging
import os
import platform
import random
import socket
import subprocess
import sys
import warnings
from warnings import warn

import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn
import torchvision
import yaml
from easydict import EasyDict
from path import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from is_ai.is_ai_song.model.base_clf import AudioClf
from is_ai.is_ai_voice.dataset.ds_mixer_clf import DatasetAudioMixer
from is_ai.losses_metrics import get_loss, get_metric
from is_ai.model.helpers import save_checkpoint, load_checkpoint
from is_ai.optimizer import get_optimizer
from is_ai.optimizer.clip import Clipper
from is_ai.scheduler import ReduceLROnPlateau
from is_ai.scheduler import get_scheduler, set_optimizer_lr
from is_ai.utils.general.computer import get_max_number_workers, choose_device
from is_ai.utils.general.custom_yaml import init_custom_yaml
from is_ai.utils.general.modifier import dict_modifier
from is_ai.utils.logger.logger import setup_logging
from is_ai.utils.torch.dataparallel import naked_model
from is_ai.is_ai_voice.trainer.trnr_clf import trainer

warnings.simplefilter('ignore', UserWarning)


def main_worker(this_gpu, n_gpus_per_node, cfg):
    cfg.this_gpu = this_gpu
    cfg.rank = cfg.node_rank * n_gpus_per_node + cfg.this_gpu
    torch.cuda.set_device(cfg.this_gpu)
    device = choose_device('cuda:%d' % cfg.this_gpu)
    cudnn.benchmark = cfg.cnn_benchmark

    print(f'DDP this_gpu={cfg.this_gpu}')
    if cfg.distributed_data_parallel:
        print(f'nccl version = {torch.cuda.nccl.version()}')
        print(f'cfg.world_size = {cfg.world_size}')
        print(f'cfg.rank = {cfg.rank}')
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://%s' % cfg.dist_address,  # init_method='env://%s' % cfg.dist_address,
            world_size=cfg.world_size,
            rank=cfg.rank)

        n_gpus_total = dist.get_world_size()
        if cfg.rank == 0:
            print(f'===> {n_gpus_total} GPUs total; batch_size={cfg.loader.train.batch_size} per GPU')
        print(f'===> Proc {dist.get_rank()}/{dist.get_world_size()}@{socket.gethostname()}', flush=True)

    print('General')
    name_save_ckpt = cfg.pop("name_save_ckpt", "valid")
    stages_trainer_sorted = cfg.pop("stages_trainer_sorted", ["train", "valid", "test"])
    base_out_folder = Path(cfg.base_out_folder)
    if cfg.engine.model.resume.save_tmp_model_fn and cfg.engine.model.resume.save_tmp_model_fn is not None:
        save_tmp_model_fn = cfg.engine.model.resume.save_tmp_model_fn
    else:
        save_tmp_model_fn = 'tmp_model.ckpt'
        logging.warning("Temporal model fn not defined. It is set as %s", str(save_tmp_model_fn))

    if cfg.rank == 0:
        if base_out_folder.isdir():
            logging.error("Output folder already exists. base_out_folder = %s", str(base_out_folder))
            raise ValueError(__name__ + ": " + run.__name__)
        base_out_folder.abspath().makedirs_p()

    if cfg.distributed_data_parallel:
        torch.distributed.barrier(device_ids=[cfg.this_gpu])

    log_dir = Path(cfg.pop("log_dir"))
    if log_dir is not None:
        log_dir.abspath().makedirs_p()

    # Logger.
    setup_logging(log_dir=log_dir, rank=this_gpu)
    logger = logging.getLogger(__name__ + ": " + run.__name__)

    # Save cfg_fn.
    logger.info('Save cfg_fn')
    json.dump(cfg, open(base_out_folder / Path(cfg.cfg_fn).stem + ".json", "w", encoding="utf8"), indent=4)

    # Info.
    logger.info("torch.__version__ = %s", torch.__version__)
    logger.info("torchvision.__version__ = %s", torchvision.__version__)
    logger.info("userame = %s", getpass.getuser())
    logger.info("hostname = %s", socket.gethostname())
    try:
        logger.info("commit = %s", subprocess.check_output(['git', 'rev-parse', 'HEAD']))
    except Exception as expt:
        logger.warning("***** Git commit was not found. *****")
        logger.warning(expt)
    logger.info("OS info = {%s, %s}", platform.system(), platform.release())

    # Device and workers.
    torch.cuda.empty_cache()
    for stage in stages_trainer_sorted:
        cfg.loader[stage]["num_workers"] = get_max_number_workers(cfg.loader[stage]["num_workers"])
        logger.info('num_workers %s = %s', stage, str(cfg.loader[stage]["num_workers"]))
    logger.info('device = %s', device)

    logger.info("Build model")
    if cfg.engine.model.name == AudioClf.__name__:
        logger.info(f"{AudioClf.__name__} building")
        model = AudioClf(cfg.engine.model)
    else:
        msg = f'Unknown model name = {cfg.engine.model.name}'
        print(msg)
        raise ValueError(msg)
    # logger.info('model = %s', model)
    input_size_test = cfg.engine.input_size_test
    input_type_test = cfg.engine.input_type_test
    logger.info("model.data_type = %s", str(model.data_type))
    logger.info("model #parameters = %s", str(model.n_parameters()))
    logger.info("model #n_parameters_grad = %s", str(model.n_parameters_grad()))
    # logger.info(f"model output size with input {input_size_test} {input_type_test} = "
    #             f"{model.size_output(input_size_test, input_type_test)}")

    logger.info('Adapt model')
    if cfg.distributed_data_parallel:
        # Apply SyncBN
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.this_gpu is not None:
            model.cuda(cfg.this_gpu)
            for stage in stages_trainer_sorted:
                cfg.loader[stage]["num_workers"] = int(cfg.loader[stage]["num_workers"] / n_gpus_per_node)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.this_gpu], find_unused_parameters=cfg.engine.model.find_unused_parameters)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = nn.parallel.DistributedDataParallel(model)
    else:
        torch.cuda.set_device(cfg.this_gpu)
        model.cuda()
    logger.info("model #parameters = %s", str(naked_model(model).n_parameters()))
    logger.info("model #n_parameters_grad = %s", str(naked_model(model).n_parameters_grad()))

    logger.info("Optimizer")
    optimizer = get_optimizer(model, cfg.optimizer)
    if cfg.engine.model.half:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    logger.info("Clipper")
    clipper = Clipper(cfg.clipper.name, args=cfg.clipper.args, rank=this_gpu)

    logger.info("Scheduler")
    lr_scheduler = get_scheduler(optimizer, cfg.scheduler)

    logger.info("Loss function")
    loss_funct = get_loss(cfg.loss.name, cfg.loss.args, rank=this_gpu)

    logger.info("Metric functions")
    metric_funct = get_metric(cfg.metric.name, cfg.metric.args)

    logger.info("Dataset and dataloader")
    loader, dataset_sampler = dict(), dict()
    dataset_sampler = dict()
    for stage in stages_trainer_sorted:
        logger.info("stage: %s", stage)
        if cfg.dataset.name == DatasetAudioMixer.__name__:
            dataset = DatasetAudioMixer(stage=stage, **cfg.dataset.get(stage), rank=cfg.rank)
        else:
            raise ValueError(f"Not implemented dataset {cfg.dataset.name}")

        if cfg.distributed_data_parallel:
            dataset_sampler[stage] = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            dataset_sampler[stage] = None

        loader[stage] = DataLoader(
            dataset, **cfg.loader.get(stage),
            shuffle=(dataset_sampler[stage] is None), sampler=dataset_sampler[stage],
            collate_fn=dataset.collate_fn)

    logger.info("Resume checkpoint")
    epoch_start, metric_patient, loss_best, metric_best = load_checkpoint(
        model, cfg.engine.model.resume.load_model_fn, lr_scheduler, optimizer, scaler, device,
        cfg.engine.model.resume.load_optimizer, cfg.engine.model.resume.load_scheduler, cfg.engine.model.resume.strict)

    if cfg.train.restart_epoch:
        epoch_start = 0

    logger.info("Restart scheduler and optimizer lr")
    if not cfg.engine.model.resume.load_scheduler:
        set_optimizer_lr(optimizer, cfg.optimizer.args.lr)
        lr_scheduler = get_scheduler(optimizer, cfg.scheduler)
        step_scheduler_at_save = cfg.scheduler.step_scheduler_at_save
        epoch_start = 0
    else:
        raise NotImplementedError("TODO: Start from scratch")

    logger.info("TensorBoard")
    if cfg.rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None

    logger.info("Training model ...")
    save_model_dir = Path(cfg.engine.model.resume.save_model_dir)
    loss = {stage: None for stage in stages_trainer_sorted}
    metric0 = {stage: None for stage in stages_trainer_sorted}
    metric1 = {stage: None for stage in stages_trainer_sorted}
    for epoch in range(epoch_start, cfg.train.max_epochs):
        lr = optimizer.param_groups[0]['lr']
        logger.info("Learning rate param_groups[0] = %s", str(lr))

        for stage in stages_trainer_sorted:
            if hasattr(dataset_sampler[stage], "set_epoch"):
                dataset_sampler[stage].set_epoch(epoch)
            loss[stage], metric0[stage], metric1[stage], model = trainer(
                epoch,
                loader[stage],
                model,
                optimizer,
                lr_scheduler,
                step_scheduler_at_save,
                loss_funct,
                metric_funct,
                stage=stage,
                scaler=scaler,
                clipper=clipper,
                writer=writer,
                n_log_interval=cfg.train.get("n_log_interval", None),
                n_save_inter_epoch=cfg.train.get("n_save_inter_epoch", None),
                grad_accum=cfg.train.get("grad_accum", 1) if stage == "train" else 1,
                this_gpu=cfg.this_gpu,
                save_tmp_model_fn=save_tmp_model_fn,
                non_blocking=cfg.train.non_blocking,
                rank=cfg.rank,
                distributed_data_parallel=cfg.distributed_data_parallel,
            )

        # Update scheduler if it is required.
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(loss["train"])
        else:
            lr_scheduler.step()

        logger.info(' ************************************************** ')
        logger.info(f' ***** lr = {optimizer.param_groups[0]["lr"]} ***** ')
        logger.info(' ************************************************** ')

        # Save.
        if loss_best is None or loss["train"] < loss_best:
            loss_best = loss["train"]
            metric_best = metric0["train"]
            metric_patient = 0
        else:
            metric_patient += 1

        if cfg.engine.model.resume.save_all and cfg.rank == 0:
            fn = f'resume_E{epoch}_L{loss["train"]:.4f}_M0:{metric0["train"]:.4f}_M1:{metric1["train"]:.4f}_L{loss[name_save_ckpt]:.4f}_M0:{metric0[name_save_ckpt]:.4f}_M1:{metric1[name_save_ckpt]:.4f}.ckpt'
            save_checkpoint(save_model_dir / fn, model=model, optimizer=optimizer, scaler=scaler,
                            lr_scheduler=lr_scheduler, epoch=epoch, loss=loss_best, metric=metric_best,
                            patient=metric_patient)

    if cfg.rank == 0:
        writer.close()

    if cfg.distributed_data_parallel:
        dist.destroy_process_group()


def run(cfg_fn):
    # Configuration.
    init_custom_yaml()
    cfg = yaml.load(open(cfg_fn), Loader=yaml.Loader)
    cfg = dict_modifier(config=cfg, modifiers="modifiers", pre_modifiers={
        "HOME": os.path.expanduser("~"), "TIME": f"{datetime.datetime.now():%Y_%m_%d_%H_%M_%S}"})
    cfg = EasyDict(cfg)
    cfg.cfg_fn = cfg_fn  # Added to save the fn of the cfg file.

    # Random Seed
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warn('***** You have chosen to seed training. This will turn on the CUDNN deterministic setting, which '
             'can slow down your training considerably! You may see unexpected behavior when restarting from '
             'checkpoints. *****')

    # Set up workers.
    n_gpus_per_node = torch.cuda.device_count()
    if cfg.distributed_data_parallel:
        cfg.world_size *= n_gpus_per_node

        # add additional argument to be able to retrieve # of processes from logs
        # and don't change initial arguments to reproduce the experiment
        cfg.number_of_processes = cfg.world_size

        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, cfg))
    else:
        # Simply call main_worker function
        cfg.number_of_processes = 1
        main_worker(0, n_gpus_per_node, cfg)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("cfg_fn", type=Path, help="Configuration file")
    args = parser.parse_args(args)
    args = vars(args)
    run(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
