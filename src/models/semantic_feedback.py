"""Backbone-agnostic C1 -> C2 -> C3 semantic reasoning prototypes."""
from __future__ import annotations

import torch
from torch import nn


class ConceptPredictor(nn.Module):
    """Predict per-region multilabel concepts from region representations."""

    def __init__(self, dim: int, hidden_dim: int, num_concepts: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, region_feat: torch.Tensor) -> torch.Tensor:
        return self.net(region_feat)


class SemanticFeedback(nn.Module):
    """Project explicit concept probabilities back into the region latent space."""

    def __init__(
        self,
        num_concepts: int,
        semantic_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
        initial_alpha: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_concepts, semantic_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_dim, latent_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(float(initial_alpha)))

    def forward(self, concept_prob: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.net(concept_prob)


class C3Predictor(nn.Module):
    """Fuse a global feature with masked mean-pooled region representations."""

    def __init__(self, dim: int = 768, hidden_dim: int = 512, num_c3: int = 10, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_c3),
        )

    def forward(
        self,
        global_feat: torch.Tensor,
        region_feat: torch.Tensor,
        region_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if region_mask is None:
            region_summary = region_feat.mean(dim=1)
        else:
            weight = region_mask.to(dtype=region_feat.dtype).unsqueeze(-1)
            region_summary = (region_feat * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.net(torch.cat((global_feat, region_summary), dim=-1))


class _HierarchyPrototypeBase(nn.Module):
    def __init__(
        self,
        dim: int = 768,
        predictor_hidden: int = 256,
        num_c1: int = 5,
        num_c2: int = 43,
        num_c3: int = 10,
        c3_hidden: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = int(dim)
        self.c1_head = ConceptPredictor(dim, predictor_hidden, num_c1, dropout)
        self.c2_head = ConceptPredictor(dim, predictor_hidden, num_c2, dropout)
        self.c3_head = C3Predictor(dim, c3_hidden, num_c3, dropout)

    def _validate_inputs(self, global_feat, region_feat, region_mask):
        if global_feat.ndim != 2 or global_feat.shape[-1] != self.dim:
            raise ValueError(f"global_feat must have shape [B,{self.dim}], got {tuple(global_feat.shape)}")
        if region_feat.ndim != 3 or region_feat.shape[0] != global_feat.shape[0] or region_feat.shape[-1] != self.dim:
            raise ValueError(
                f"region_feat must have shape [B,R,{self.dim}] with matching B, "
                f"got {tuple(region_feat.shape)}"
            )
        if region_mask is not None and region_mask.shape != region_feat.shape[:2]:
            raise ValueError(
                f"region_mask must have shape {tuple(region_feat.shape[:2])}, "
                f"got {tuple(region_mask.shape)}"
            )


class AuxiliaryConceptPrototype(_HierarchyPrototypeBase):
    """Control B: auxiliary C1/C2 heads without semantic feedback."""

    def forward(self, global_feat, region_feat, region_mask=None):
        self._validate_inputs(global_feat, region_feat, region_mask)
        r0 = region_feat
        c1_logits = self.c1_head(r0)
        c2_logits = self.c2_head(r0)
        c3_logits = self.c3_head(global_feat, r0, region_mask)
        return {
            "c1_logits": c1_logits,
            "c2_logits": c2_logits,
            "c3_logits": c3_logits,
            "c1_prob": torch.sigmoid(c1_logits),
            "c2_prob": torch.sigmoid(c2_logits),
            "r0": r0,
            "r1": r0,
            "r2": r0,
        }


class SemanticFeedbackPrototype(_HierarchyPrototypeBase):
    """Prototype C: C1 and C2 probabilities explicitly update later latent states."""

    def __init__(
        self,
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
    ):
        super().__init__(dim, predictor_hidden, num_c1, num_c2, num_c3, c3_hidden, dropout)
        self.stop_gradient = bool(stop_gradient)
        self.c1_feedback = SemanticFeedback(
            num_c1, semantic_hidden, dim, dropout, initial_alpha
        )
        self.c2_feedback = SemanticFeedback(
            num_c2, semantic_hidden, dim, dropout, initial_alpha
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def _feedback_input(self, probability: torch.Tensor) -> torch.Tensor:
        return probability.detach() if self.stop_gradient else probability

    def forward(self, global_feat, region_feat, region_mask=None):
        self._validate_inputs(global_feat, region_feat, region_mask)
        r0 = region_feat

        c1_logits = self.c1_head(r0)
        c1_prob = torch.sigmoid(c1_logits)
        r1 = self.norm1(r0 + self.c1_feedback(self._feedback_input(c1_prob)))

        c2_logits = self.c2_head(r1)
        c2_prob = torch.sigmoid(c2_logits)
        r2 = self.norm2(r1 + self.c2_feedback(self._feedback_input(c2_prob)))

        c3_logits = self.c3_head(global_feat, r2, region_mask)
        return {
            "c1_logits": c1_logits,
            "c2_logits": c2_logits,
            "c3_logits": c3_logits,
            "c1_prob": c1_prob,
            "c2_prob": c2_prob,
            "r0": r0,
            "r1": r1,
            "r2": r2,
        }
