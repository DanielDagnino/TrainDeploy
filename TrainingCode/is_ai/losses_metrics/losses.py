#!/usr/bin/env python
import inspect
import logging

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional
from torch.nn.modules.loss import BCEWithLogitsLoss


class MemoryL2EmbeddingLoss(Module):
    def __init__(self, margin_pos: float = 0, margin_neg: float = 1, memory: int = 1024, p_keep: float = 0.9,
                 rank: int = 0) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        self.memory = memory
        self.margin_pos = margin_pos
        self.margin_neg = margin_neg
        self.rank = rank
        self.emb_mem = None
        self.lbl_mem = None
        self.p_keep = p_keep

    def forward(self, embeddings: Tensor, labels: Tensor, add_to_mem: Tensor) -> Tensor:
        bs = embeddings.shape[0]
        if self.emb_mem is not None:
            # Remove repeated indexes (can happen because the HN miner).
            repeated = torch.isin(self.lbl_mem, labels)
            self.emb_mem = self.emb_mem[~repeated]
            self.lbl_mem = self.lbl_mem[~repeated]

            # Cut to self.memory
            self.emb_mem = self.emb_mem[:(self.memory - bs)]
            self.lbl_mem = self.lbl_mem[:(self.memory - bs)]
            idx_ref = -torch.ones(len(self.lbl_mem), dtype=self.lbl_mem.dtype, device=self.lbl_mem.device)

            # Add to ref
            embeddings_ref = torch.cat([embeddings.detach(), self.emb_mem], dim=0)
            labels_ref = torch.cat([labels.detach(), self.lbl_mem], dim=0)
            idx_ref = torch.cat(
                [torch.arange(len(labels), dtype=labels.dtype, device=labels.device), idx_ref], dim=0)
        else:
            embeddings_ref = embeddings.detach()
            labels_ref = labels.detach()
            idx_ref = torch.arange(len(labels), dtype=labels.dtype, device=labels.device)

        fraction = len(embeddings_ref) / self.memory
        if self.rank == 0 and fraction < 0.98:
            self.logger.info(f"Fraction memory loaded = {fraction}")

        # Loss
        nb, nd = embeddings.shape
        loss_pos = torch.tensor(0., dtype=embeddings.dtype, device=embeddings.device)
        loss_neg = torch.tensor(0., dtype=embeddings.dtype, device=embeddings.device)
        for ib in range(nb):
            # ap and an
            idx1 = torch.arange(len(idx_ref), dtype=idx_ref.dtype, device=idx_ref.device)
            mask_eq = (labels[ib] == labels_ref) & (idx1[ib] != idx_ref)
            mask_ne = (labels[ib] != labels_ref) & (idx1[ib] != idx_ref)
            emb_ref_eq = embeddings_ref[mask_eq]
            emb_ref_ne = embeddings_ref[mask_ne]

            # loss
            dst_ap = torch.pow(embeddings[ib] - emb_ref_eq, 2).sum(1)  # .sqrt_()
            dst_an = torch.pow(embeddings[ib] - emb_ref_ne, 2).sum(1)  # .sqrt_()
            loss_ap = dst_ap - self.margin_pos
            loss_an = functional.relu(self.margin_neg - dst_an)
            # min dist (m=1): dst_an=0 => loss_an = 1
            # min dist (m=1): dst_an=0.99 => loss_an = 1-0.99
            # max dist (m=1): dst_an=1 or 2 or 4 => loss_an = 0

            # reducer
            mask_pos = loss_ap > 0
            mask_neg = loss_an > 0
            loss_pos += loss_ap[mask_pos].sum() / (mask_pos.sum() + 1.e-6)
            loss_neg += loss_an[mask_neg].sum() / (mask_neg.sum() + 1.e-6)
        loss = (loss_pos + loss_neg) / nb

        # Keep only add_to_mem.
        emb_to_add = embeddings[add_to_mem].detach()
        lbl_to_add = labels[add_to_mem].detach()

        emb_to_add, lbl_to_add = self.rmv_bad_against_all(emb_to_add, lbl_to_add)

        # Memorize
        if self.emb_mem is not None:
            self.emb_mem = torch.cat(
                [emb_to_add, self.emb_mem[:(self.memory - len(emb_to_add))]], dim=0)
            self.lbl_mem = torch.cat(
                [lbl_to_add, self.lbl_mem[:(self.memory - len(lbl_to_add))]], dim=0)
        else:
            self.emb_mem = emb_to_add
            self.lbl_mem = lbl_to_add

        return loss

    def rmv_bad_against_all(self, emb_to_add, lbl_to_add):
        if self.emb_mem is not None and self.p_keep < 1:
            dst_to_add = torch.zeros(len(emb_to_add), dtype=torch.float, device=emb_to_add.device)
            for ib in range(len(emb_to_add)):
                dst = torch.pow(emb_to_add[ib] - self.emb_mem, 2).sum(1).sum(0)  # .sqrt_()
                dst_to_add[ib] = dst.item()
            idx = torch.argsort(-dst_to_add)[:int(self.p_keep * len(dst_to_add))]
            emb_to_add = emb_to_add[idx]
            lbl_to_add = lbl_to_add[idx]
        return emb_to_add, lbl_to_add


class BinaryCrossEntropyWithLogits(Module):
    def __init__(self, weight_0=None, device: torch.device = "cuda:0", rank: int = 0) -> None:
        super().__init__()
        if weight_0 is not None:
            self.loss = BCEWithLogitsLoss(reduction='none').to(device)
        else:
            self.loss = BCEWithLogitsLoss(reduction='mean').to(device)
        self.weight_0 = weight_0

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        outputs = outputs.view(-1)
        targets = targets.view(-1).type_as(outputs)
        if self.weight_0 is not None:
            weight = torch.ones_like(targets)
            weight[targets < 1] = self.weight_0
            loss = weight * self.loss(outputs, targets)
            return loss.mean()
        return self.loss(outputs, targets)
