"""Frozen CXR-CLIP global and bbox-aware region feature extraction."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torchvision.ops import roi_align

from src.models.cxr_c3_baseline import CXRClipSwinEncoder


class BBoxRegionPooler(nn.Module):
    """Pool spatial features for bbox regions and use global fallback otherwise."""

    def __init__(self, spatial_dim: int, output_dim: int = 768, num_regions: int = 32):
        super().__init__()
        self.output_dim = int(output_dim)
        self.num_regions = int(num_regions)
        self.projection = (
            nn.Identity() if spatial_dim == output_dim else nn.Linear(spatial_dim, output_dim)
        )
        self.region_embedding = nn.Parameter(torch.empty(num_regions, output_dim))
        nn.init.trunc_normal_(self.region_embedding, std=0.02)

    @staticmethod
    def _validate_boxes(region_bboxes, bbox_mask, image_size):
        width, height = image_size
        if region_bboxes.ndim != 3 or region_bboxes.shape[-1] != 4:
            raise ValueError(f"region_bboxes must have shape [B,R,4], got {tuple(region_bboxes.shape)}")
        if bbox_mask.shape != region_bboxes.shape[:2]:
            raise ValueError(
                f"bbox_mask must have shape {tuple(region_bboxes.shape[:2])}, "
                f"got {tuple(bbox_mask.shape)}"
            )
        active = region_bboxes[bbox_mask]
        if active.numel() == 0:
            return
        valid = (
            (active[:, 0] >= 0) & (active[:, 1] >= 0)
            & (active[:, 0] < active[:, 2]) & (active[:, 1] < active[:, 3])
            & (active[:, 2] <= width) & (active[:, 3] <= height)
        )
        if not bool(valid.all()):
            raise ValueError("Every bbox_mask=True box must satisfy 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H")

    def forward(self, spatial_feat, global_feat, region_bboxes, bbox_mask, image_size):
        batch_size, channels, feature_height, feature_width = spatial_feat.shape
        image_width, image_height = image_size
        if global_feat.shape != (batch_size, self.output_dim):
            raise ValueError(
                f"global_feat must have shape {(batch_size, self.output_dim)}, "
                f"got {tuple(global_feat.shape)}"
            )
        if region_bboxes.shape[:2] != (batch_size, self.num_regions):
            raise ValueError(
                f"Expected {self.num_regions} regions, got shape {tuple(region_bboxes.shape)}"
            )
        if feature_width * image_height != feature_height * image_width:
            raise ValueError("ROIAlign requires matching image and feature-map aspect ratios")

        bbox_mask = bbox_mask.to(device=spatial_feat.device, dtype=torch.bool)
        region_bboxes = region_bboxes.to(device=spatial_feat.device, dtype=spatial_feat.dtype)
        self._validate_boxes(region_bboxes, bbox_mask, image_size)

        # All regions receive a usable visual representation. Regions without
        # geometry fall back to the official global feature plus anatomy identity.
        region_feat = global_feat[:, None, :].expand(-1, self.num_regions, -1).clone()
        active_indices = bbox_mask.nonzero(as_tuple=False)
        if active_indices.numel():
            active_boxes = region_bboxes[bbox_mask]
            rois = torch.cat(
                (active_indices[:, :1].to(active_boxes.dtype), active_boxes), dim=1
            )
            pooled = roi_align(
                spatial_feat,
                rois,
                output_size=(1, 1),
                spatial_scale=feature_width / image_width,
                sampling_ratio=-1,
                aligned=True,
            ).flatten(1)
            region_feat[bbox_mask] = self.projection(pooled).to(region_feat.dtype)

        region_feat = region_feat + self.region_embedding.unsqueeze(0)
        region_valid = torch.ones(
            batch_size, self.num_regions, dtype=torch.bool, device=region_feat.device
        )
        return region_feat, region_valid


class FrozenCXRCLIPFeatureExtractor(nn.Module):
    """Stable image/bbox adapter for the B and C hierarchy reasoning heads.

    Input contract:
        img: [B,3,224,224]
        region_bboxes: [B,32,4] in transformed image pixel coordinates
        bbox_mask: [B,32], where True means real geometry is available

    Output contract:
        global_feat: [B,768]
        region_feat: [B,32,768]
        region_valid: [B,32], including global-fallback regions
    """

    STAGE_SPECS = {
        "final": {"hidden_index": None, "channels": 768, "resolution": 7},
        "penultimate": {"hidden_index": 2, "channels": 384, "resolution": 14},
    }

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        spatial_stage: str = "penultimate",
        image_size: int = 224,
        num_regions: int = 32,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        if spatial_stage not in self.STAGE_SPECS:
            raise ValueError(f"spatial_stage must be one of {tuple(self.STAGE_SPECS)}")
        if backbone is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when backbone is not supplied")
            backbone = CXRClipSwinEncoder(checkpoint_path)
        if int(backbone.out_dim) != 768:
            raise ValueError(f"Expected a 768-d CXR-CLIP SwinTiny backbone, got {backbone.out_dim}")

        self.backbone = backbone
        self.spatial_stage = spatial_stage
        self.image_size = int(image_size)
        self.num_regions = int(num_regions)
        spec = self.STAGE_SPECS[spatial_stage]
        self.region_pooler = BBoxRegionPooler(spec["channels"], 768, num_regions)
        self._freeze_backbone()

    def _freeze_backbone(self):
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError(f"Spatial token count must be square, got {tokens.shape[1]}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)

    def forward(self, img, region_bboxes, bbox_mask):
        if img.ndim != 4 or img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                f"img must have shape [B,3,{self.image_size},{self.image_size}], "
                f"got {tuple(img.shape)}"
            )
        if region_bboxes.shape != (img.shape[0], self.num_regions, 4):
            raise ValueError(
                f"region_bboxes must have shape {(img.shape[0], self.num_regions, 4)}, "
                f"got {tuple(region_bboxes.shape)}"
            )

        with torch.no_grad():
            output = self.backbone.encoder(
                pixel_values=img, output_hidden_states=True, return_dict=True
            )
            # Preserve official CXR-CLIP encode_image behavior exactly, including
            # its use of token 0 for the Swin global representation.
            global_feat = output.last_hidden_state[:, 0]
            if self.spatial_stage == "final":
                spatial_tokens = output.last_hidden_state
            else:
                spatial_tokens = output.hidden_states[
                    self.STAGE_SPECS[self.spatial_stage]["hidden_index"]
                ]
            spatial_feat = self._tokens_to_map(spatial_tokens)

        region_feat, region_valid = self.region_pooler(
            spatial_feat,
            global_feat,
            region_bboxes,
            bbox_mask,
            image_size=(self.image_size, self.image_size),
        )
        return {
            "global_feat": global_feat,
            "region_feat": region_feat,
            "region_valid": region_valid,
        }
