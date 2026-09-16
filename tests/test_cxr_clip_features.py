from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.cxr_clip_features import BBoxRegionPooler, FrozenCXRCLIPFeatureExtractor


class FakeSwin(nn.Module):
    def forward(self, pixel_values, output_hidden_states, return_dict):
        batch = pixel_values.shape[0]
        device = pixel_values.device
        penultimate = torch.randn(batch, 196, 384, device=device)
        final = torch.randn(batch, 49, 768, device=device)
        return SimpleNamespace(
            last_hidden_state=final,
            hidden_states=(
                torch.empty(batch, 3136, 96, device=device),
                torch.empty(batch, 784, 192, device=device),
                penultimate,
                final,
                final,
            ),
        )


class FakeCXRCLIPBackbone(nn.Module):
    out_dim = 768

    def __init__(self):
        super().__init__()
        self.encoder = FakeSwin()
        self.sentinel = nn.Parameter(torch.ones(()))


def sample_geometry(batch=2):
    boxes = torch.zeros(batch, 32, 4)
    mask = torch.zeros(batch, 32, dtype=torch.bool)
    boxes[:, 0] = torch.tensor([0, 0, 224, 224])
    boxes[:, 1] = torch.tensor([20, 30, 100, 160])
    mask[:, :2] = True
    return boxes, mask


@pytest.mark.parametrize("stage", ["final", "penultimate"])
def test_extractor_contract_and_frozen_backbone(stage):
    backbone = FakeCXRCLIPBackbone()
    extractor = FrozenCXRCLIPFeatureExtractor(
        spatial_stage=stage, backbone=backbone
    ).train()
    boxes, mask = sample_geometry()
    output = extractor(torch.randn(2, 3, 224, 224), boxes, mask)

    assert output["global_feat"].shape == (2, 768)
    assert output["region_feat"].shape == (2, 32, 768)
    assert output["region_valid"].shape == (2, 32)
    assert output["region_valid"].all()
    assert not output["global_feat"].requires_grad
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
    assert not backbone.training


def test_missing_bbox_uses_global_feature_plus_region_embedding():
    pooler = BBoxRegionPooler(spatial_dim=4, output_dim=4, num_regions=3)
    pooler.region_embedding.data.zero_()
    spatial = torch.randn(1, 4, 7, 7)
    global_feat = torch.randn(1, 4)
    boxes = torch.tensor([[[0, 0, 224, 224], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float32)
    mask = torch.tensor([[True, False, False]])

    region_feat, region_valid = pooler(spatial, global_feat, boxes, mask, (224, 224))

    assert torch.equal(region_feat[:, 1:], global_feat[:, None, :].expand(-1, 2, -1))
    assert region_valid.all()


def test_bbox_mask_and_region_valid_have_distinct_semantics():
    extractor = FrozenCXRCLIPFeatureExtractor(backbone=FakeCXRCLIPBackbone())
    boxes, bbox_mask = sample_geometry(batch=1)
    output = extractor(torch.randn(1, 3, 224, 224), boxes, bbox_mask)
    assert bbox_mask.sum() == 2
    assert output["region_valid"].sum() == 32


def test_invalid_active_bbox_is_rejected():
    pooler = BBoxRegionPooler(spatial_dim=4, output_dim=4, num_regions=1)
    with pytest.raises(ValueError, match="bbox_mask=True"):
        pooler(
            torch.randn(1, 4, 7, 7),
            torch.randn(1, 4),
            torch.tensor([[[20, 30, 10, 40]]], dtype=torch.float32),
            torch.ones(1, 1, dtype=torch.bool),
            (224, 224),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pooler_accepts_cpu_geometry_with_cuda_features():
    pooler = BBoxRegionPooler(spatial_dim=4, output_dim=4, num_regions=1).cuda()
    region_feat, region_valid = pooler(
        torch.randn(1, 4, 7, 7, device="cuda"),
        torch.randn(1, 4, device="cuda"),
        torch.tensor([[[0, 0, 224, 224]]], dtype=torch.float32),
        torch.ones(1, 1, dtype=torch.bool),
        (224, 224),
    )
    assert region_feat.is_cuda
    assert region_valid.is_cuda
