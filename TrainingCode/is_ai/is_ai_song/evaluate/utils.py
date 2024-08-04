import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_recall_curve, roc_curve, \
    average_precision_score

from is_ai.is_ai_song.evaluate.metrics import get_tpr_at_fpr_perc


def compute_metrics(gt, pred, scores):
    """
    Compute metrics.

    Args:
        gt: ground truth
        pred: predictions
        scores: scores
    Returns
        dict: metric results
    """

    # confusion matrix
    conf = confusion_matrix(np.copy(gt), np.copy(pred))
    tn, fn = conf[0, 0], conf[1, 0]
    fp, tp = conf[0, 1], conf[1, 1]

    pp = tp + fp
    cp = tp + fn
    cn = tn + fp

    tpr = tp / cp
    fpr = fp / cn

    # sklearn stats
    avrg_precs = average_precision_score(np.copy(gt), np.copy(scores))
    precision, recall, f1, _ = precision_recall_fscore_support(np.copy(gt), np.copy(pred), average='binary')

    # fpr vs tpr curve
    fpr_values, tpr_values, roc_thrsh = roc_curve(np.copy(gt), np.copy(scores))

    # TPRatFPRExp
    TPRatFPRExp2 = get_tpr_at_fpr_perc(fpr_values, tpr_values, perc=1.e-2)
    TPRatFPRExp3 = get_tpr_at_fpr_perc(fpr_values, tpr_values, perc=1.e-3)

    # PR curve
    skl_precs_thrsh, skl_recall_thresholds, skl_thresholds = precision_recall_curve(np.copy(gt), np.copy(scores))

    res_json = {
        "samples": pred.shape[0],
        "CP": int(cp),
        "CN": int(cn),
        "PP": int(pp),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "avg_precision": float(avrg_precs),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),

        "TPRatFPRExp2": float(TPRatFPRExp2),
        "TPRatFPRExp3": float(TPRatFPRExp3),

        "tpr_fpr_curve": {
            "tpr": tpr_values.tolist(),
            "fpr": fpr_values.tolist(),
            "thresholds": roc_thrsh.tolist()
        },
        "PR_curve": {
            "precisions": skl_precs_thrsh.tolist(),
            "recalls": skl_recall_thresholds.tolist(),
            "thresholds": skl_thresholds.tolist()
        }
    }
    return res_json
