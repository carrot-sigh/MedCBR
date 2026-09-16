"""Masked multilabel loss and metrics for hierarchy V1 C3 diagnoses."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score


def masked_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Compute BCE only for explicit C3 assertions; unknown -1 entries are ignored."""
    if logits.shape != target.shape or logits.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, target={target.shape}, mask={mask.shape}"
        )
    mask = mask.bool()
    safe_target = torch.where(mask, target, torch.zeros_like(target)).float()
    raw_loss = F.binary_cross_entropy_with_logits(logits, safe_target, reduction="none")
    return (raw_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _masked_class_data(probabilities, targets, masks, class_index):
    active = masks[:, class_index].astype(bool)
    return targets[active, class_index].astype(np.int64), probabilities[active, class_index]


def tune_f1_thresholds(
    probabilities: np.ndarray, targets: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """Tune one threshold per class on validation data only."""
    thresholds = np.full(probabilities.shape[1], np.nan, dtype=np.float64)
    for class_index in range(probabilities.shape[1]):
        y_true, y_score = _masked_class_data(probabilities, targets, masks, class_index)
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, candidates = precision_recall_curve(y_true, y_score)
        if not len(candidates):
            continue
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        thresholds[class_index] = float(candidates[int(np.nanargmax(f1))])
    return thresholds


def masked_multilabel_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    class_names: Sequence[str],
    thresholds: np.ndarray | None = None,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    masks = np.asarray(masks, dtype=bool)
    if probabilities.shape != targets.shape or probabilities.shape != masks.shape:
        raise ValueError("probabilities, targets and masks must have identical shapes")
    if probabilities.shape[1] != len(class_names):
        raise ValueError("class_names length does not match prediction width")
    if thresholds is None:
        thresholds = np.full(probabilities.shape[1], 0.5, dtype=np.float64)

    per_class = []
    for class_index, class_name in enumerate(class_names):
        y_true, y_score = _masked_class_data(probabilities, targets, masks, class_index)
        positive = int((y_true == 1).sum())
        negative = int((y_true == 0).sum())
        auroc = float("nan")
        auprc = float("nan")
        if positive and negative:
            auroc = float(roc_auc_score(y_true, y_score))
            auprc = float(average_precision_score(y_true, y_score))
        f1 = float("nan")
        if len(y_true) and np.isfinite(thresholds[class_index]):
            prediction = (y_score >= thresholds[class_index]).astype(np.int64)
            f1 = float(f1_score(y_true, prediction, zero_division=0))
        per_class.append(
            {
                "diagnosis": class_name,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "threshold": float(thresholds[class_index]),
                "positive": positive,
                "negative": negative,
                "explicit": int(len(y_true)),
            }
        )

    def macro(key):
        values = [row[key] for row in per_class if not math.isnan(row[key])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "macro_auroc": macro("auroc"),
        "macro_auprc": macro("auprc"),
        "macro_f1": macro("f1"),
        "evaluable_auroc_classes": sum(not math.isnan(row["auroc"]) for row in per_class),
        "per_class": per_class,
    }
