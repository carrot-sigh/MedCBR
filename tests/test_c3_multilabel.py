import numpy as np
import torch

from src.utils.c3_multilabel import (
    masked_bce_with_logits,
    masked_multilabel_metrics,
    tune_f1_thresholds,
)


def test_unknown_targets_do_not_affect_masked_bce():
    logits = torch.tensor([[0.0, 100.0, -100.0], [0.0, -100.0, 100.0]])
    targets = torch.tensor([[1.0, -1.0, -1.0], [0.0, -1.0, -1.0]])
    masks = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.bool)
    expected = torch.tensor(0.69314718)
    assert torch.allclose(masked_bce_with_logits(logits, targets, masks), expected)


def test_metrics_and_thresholds_use_only_explicit_labels():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.3], [0.1, 0.7]])
    targets = np.array([[1, -1], [0, 1], [1, 0], [0, 1]])
    masks = np.array([[1, 0], [1, 1], [1, 1], [1, 1]], dtype=bool)
    thresholds = tune_f1_thresholds(probabilities, targets, masks)
    metrics = masked_multilabel_metrics(probabilities, targets, masks, ["a", "b"], thresholds)
    assert metrics["macro_auroc"] == 1.0
    assert metrics["macro_auprc"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_single_polarity_class_has_undefined_ranking_metrics():
    probabilities = np.array([[0.8], [0.9]])
    targets = np.array([[1], [1]])
    masks = np.ones_like(targets, dtype=bool)
    thresholds = tune_f1_thresholds(probabilities, targets, masks)
    metrics = masked_multilabel_metrics(probabilities, targets, masks, ["all_positive"], thresholds)
    assert np.isnan(thresholds[0])
    assert np.isnan(metrics["per_class"][0]["auroc"])
    assert np.isnan(metrics["per_class"][0]["auprc"])
    assert np.isnan(metrics["per_class"][0]["f1"])
