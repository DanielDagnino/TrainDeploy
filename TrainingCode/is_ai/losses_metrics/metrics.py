import inspect
import logging
from typing import Tuple

import torch
from torch import Tensor


class CosineEmbedding:
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, embeddings: Tensor, labels: Tensor) -> Tuple[float, float, int, int]:
        nb, nd = embeddings.shape
        n_pos, n_neg = 0, 0
        metric0, metric1 = 0, 0
        for ib in range(nb):
            # Reference vector
            emb1 = embeddings[ib, :].reshape(1, nd)

            # Find matches to reference
            mask_eq = torch.eq(labels[ib], labels)
            mask_ne = ~mask_eq

            # Do not consider same vector
            mask_eq[ib] = False
            mask_ne[ib] = False

            # Count pos and neg
            n_pos += mask_eq.sum()
            n_neg += mask_ne.sum()

            # Metric
            metric1 += (emb1 * embeddings[mask_eq]).sum(1).sum(0).item()
            metric0 += (emb1 * embeddings[mask_ne]).sum(1).sum(0).item()

        return metric1 / (n_pos + 1.e-9), metric0 / (n_neg + 1.e-9), n_pos, n_neg


class BinaryAccuracy:
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.logger = logging.getLogger(__name__ + ": " + inspect.currentframe().f_code.co_name)
        if not 0 < threshold < 1:
            self.logger.error("threshold must be in the range (0, 1), however threshold = %s", str(threshold))
            raise ValueError(__name__ + ": " + self.__class__.__qualname__)
        self.threshold = threshold

    def __call__(self, scores: Tensor, targets: Tensor) -> Tuple[float, int, float, int, float, int]:
        scores = scores.view(-1)
        targets = targets.view(-1)
        n_total = len(scores)

        acc = ((scores > self.threshold) == targets).type(torch.float).mean().item()

        mask = targets == 1
        correct1 = ((scores[mask] > self.threshold) == targets[mask]).type(torch.float).sum().item()
        correct0 = ((scores[~mask] > self.threshold) == targets[~mask]).type(torch.float).sum().item()

        n_pos = targets.sum()
        n_neg = n_total - n_pos
        return acc, n_total, correct0 / (n_neg + 1.e-6), n_neg, correct1 / (n_pos + 1.e-6), n_pos
