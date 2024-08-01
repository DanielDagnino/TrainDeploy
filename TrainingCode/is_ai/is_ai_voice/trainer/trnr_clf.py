#!/usr/bin/env python
import gc
import inspect
import logging
import time
from typing import Dict, Optional, Callable

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.distributed import ReduceOp
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from is_ai.model.helpers import save_checkpoint
from is_ai.losses_metrics import AverageMeter
from is_ai.optimizer import RETURN_optimizer_builder
from is_ai.optimizer.clip import Clipper
from is_ai.scheduler import RETURN_scheduler_builder


def trainer(epoch: int,
            data_loader: DataLoader,
            model: Module,
            optimizer: RETURN_optimizer_builder,
            lr_scheduler: RETURN_scheduler_builder,
            step_scheduler_at_save: bool,
            loss_funct: Module,
            metric_funct: Optional[Callable],
            scaler: GradScaler = None,
            writer: SummaryWriter = None,
            stage: str = "train",
            clipper: Clipper = None,
            grad_accum: int = 1,
            this_gpu: int = 0,
            rank: int = 0,
            n_log_interval: int = 100,
            n_save_inter_epoch: int = 100,
            save_tmp_model_fn: str = None,
            non_blocking: bool = True,
            distributed_data_parallel: bool = False
            ) -> (float, Dict[str, float], Module):
    logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)

    # Select whether trainable or not.
    if stage == "train":
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    # Meters.
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()
    metric_meter0, metric_meter1 = AverageMeter(accept_zero_samples=True), AverageMeter(accept_zero_samples=True)

    # Loop over mini-batches.
    logger.info(f"{stage} Ep={epoch}")
    it = None
    start_time = time.time()
    for it, batch in enumerate(data_loader):
        path0, path1, segs0, segs1, labels0, labels1 = batch
        segs = torch.cat(
            [segs0, segs1], dim=0
        ).detach().requires_grad_(requires_grad=False).cuda(this_gpu, non_blocking=non_blocking)
        labels = torch.cat(
            [labels0, labels1], dim=0
        ).long().detach().requires_grad_(requires_grad=False).cuda(this_gpu, non_blocking=non_blocking)

        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float32 if scaler is None else torch.float16):
                out = model(segs)
                loss = loss_funct(out, labels)
        else:
            out = model(segs)
            loss = loss_funct(out, labels)

        if torch.isnan(loss):
            msg = f"NaN loss found at it={it}"
            logger.error(msg)
            raise ValueError(msg)

        if stage == "train":
            loss = (1. / grad_accum) * loss
            if scaler is not None:
                scaler.scale(loss).backward()
                if (it + 1) % grad_accum == 0:
                    if clipper.is_not_null():
                        scaler.unscale_(optimizer)
                        clipper.apply_to_grad(model)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (it + 1) % grad_accum == 0:
                    if clipper.is_not_null():
                        clipper.apply_to_grad(model)
                    optimizer.step()
                    optimizer.zero_grad()

        loss_meter.update(grad_accum * loss.item(), segs0.shape[0])
        if metric_funct is not None:
            with torch.no_grad():
                metric, cnt, metric_0, n_0, metric_1, n_1 = metric_funct(out, labels)
                metric_meter.update(metric, cnt)
                metric_meter0.update(metric_0, n_0)
                metric_meter1.update(metric_1, n_1)

        # Intermediate results.
        if (it + 1) % n_log_interval == 0:
            time_elapse = 1000 * (time.time() - start_time)
            logger.info(f" DL {(it + 1) / len(data_loader):.3f} | L {loss_meter.val:.5f} | LT {loss_meter.avg:.5f} | "
                        f"{time_elapse:.5}ms cuda:{this_gpu}")
            logger.info(
                f"          | MA01 {metric_meter.val: .3f} {metric_meter0.val: .3f} {metric_meter1.val: .3f} | "
                f"MT {metric_meter.avg: .3f} {metric_meter0.avg: .3f} {metric_meter1.avg: .3f}")
            start_time = time.time()

            if writer is not None:
                step = it + epoch * len(data_loader)
                writer.add_scalar(f"Loss/{stage}", loss_meter.val, global_step=step)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], global_step=step)

        if (it + 1) % n_save_inter_epoch == 0:
            if step_scheduler_at_save:
                lr_scheduler.step()
                logger.info(' ************************************************** ')
                logger.info(f' ***** lr = {optimizer.param_groups[0]["lr"]} ***** ')
                logger.info(' ************************************************** ')
            if rank == 0:
                logger.info(f'Saving temporal model {save_tmp_model_fn}')
                save_checkpoint(save_tmp_model_fn, model=model, optimizer=optimizer, scaler=scaler,
                                lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)
                save_checkpoint(save_tmp_model_fn[:-5] + f"_it={it}.ckpt", model=model, optimizer=optimizer,
                                scaler=scaler, lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)

            del path0, path1, segs0, segs1, labels0, labels1
            del labels, segs
            del loss, out
            gc.collect()
            torch.cuda.memory_reserved()

    # Final results.
    logger.info(f" DL {(it + 1) / len(data_loader):.3f} | L {loss_meter.val:.5f} | LT {loss_meter.avg:.5f}")
    logger.info(
        f"         | MA01 {metric_meter.val: .3f} {metric_meter0.val: .3f} {metric_meter1.val: .3f} | MT {metric_meter.avg: .3f} {metric_meter0.avg: .3f} {metric_meter1.avg: .3f}")

    logger.info(f'Saving temporal model {save_tmp_model_fn}')
    save_checkpoint(save_tmp_model_fn, model=model, optimizer=optimizer, scaler=scaler,
                    lr_scheduler=lr_scheduler, epoch=epoch, loss=0, metric=0, patient=0)

    # Gather metric results.
    metric0_all = torch.tensor(metric_meter0.sum)
    metric1_all = torch.tensor(metric_meter1.sum)
    count0_all = torch.tensor(metric_meter0.count)
    count1_all = torch.tensor(metric_meter1.count)
    if distributed_data_parallel:
        dist.all_reduce(metric0_all, op=ReduceOp.SUM)
        dist.all_reduce(metric1_all, op=ReduceOp.SUM)
        dist.all_reduce(count0_all, op=ReduceOp.SUM)
        dist.all_reduce(count1_all, op=ReduceOp.SUM)
    metric0_all = metric0_all.item() / count0_all.item()
    metric1_all = metric1_all.item() / count1_all.item()

    return loss_meter.avg, metric0_all, metric1_all, model
