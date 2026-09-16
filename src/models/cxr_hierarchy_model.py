"""Composition boundary for frozen CXR-CLIP and hierarchy reasoning controls."""
from __future__ import annotations

from pathlib import Path

from torch import nn

from src.models.cxr_clip_features import FrozenCXRCLIPFeatureExtractor
from src.models.semantic_feedback import AuxiliaryConceptPrototype, SemanticFeedbackPrototype


class FrozenCXRCLIPHierarchyModel(nn.Module):
    """Run one of the controlled B/C reasoning variants on shared visual features."""

    VARIANTS = ("b", "c", "c_zero", "c_shuffle")

    def __init__(
        self,
        checkpoint_path: str | Path,
        variant: str,
        spatial_stage: str = "penultimate",
        dim: int = 768,
        predictor_hidden: int = 256,
        semantic_hidden: int = 128,
        num_c1: int = 5,
        num_c2: int = 43,
        num_c3: int = 10,
        c3_hidden: int = 512,
        dropout: float = 0.1,
        stop_gradient: bool = True,
        initial_alpha: float = 0.1,
        feature_extractor: nn.Module | None = None,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        if dim != 768:
            raise ValueError("Frozen CXR-CLIP feature contract requires dim=768")
        self.variant = variant
        self.feature_extractor = feature_extractor or FrozenCXRCLIPFeatureExtractor(
            checkpoint_path=checkpoint_path,
            spatial_stage=spatial_stage,
        )
        common = {
            "dim": dim,
            "predictor_hidden": predictor_hidden,
            "num_c1": num_c1,
            "num_c2": num_c2,
            "num_c3": num_c3,
            "c3_hidden": c3_hidden,
            "dropout": dropout,
        }
        if variant == "b":
            self.reasoning = AuxiliaryConceptPrototype(**common)
        else:
            feedback_mode = {
                "c": "semantic",
                "c_zero": "zero",
                "c_shuffle": "shuffle",
            }[variant]
            self.reasoning = SemanticFeedbackPrototype(
                **common,
                semantic_hidden=semantic_hidden,
                stop_gradient=stop_gradient,
                initial_alpha=initial_alpha,
                feedback_mode=feedback_mode,
            )

    def forward(self, img, region_bboxes, bbox_mask):
        visual = self.feature_extractor(img, region_bboxes, bbox_mask)
        output = self.reasoning(
            visual["global_feat"], visual["region_feat"], visual["region_valid"]
        )
        output["region_valid"] = visual["region_valid"]
        return output
