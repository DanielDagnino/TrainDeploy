import numpy as np


def get_tpr_at_fpr_perc(fpr, tpr, perc=1.e-2):
    """FPr below a TPr of perc."""
    idx = np.lexsort((tpr, fpr))
    fpr, tpr = fpr[idx], tpr[idx]
    idx_cut = int(np.argmax(fpr >= perc))
    if idx_cut > 0:
        return tpr[idx_cut - 1]
    return 0
