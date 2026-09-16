import pytest
import torch
from torch import nn

from src.models.cxr_hierarchy_model import FrozenCXRCLIPHierarchyModel


class FakeFeatureExtractor(nn.Module):
    def forward(self, img, region_bboxes, bbox_mask):
        batch = img.shape[0]
        return {
            "global_feat": torch.randn(batch, 768),
            "region_feat": torch.randn(batch, 32, 768),
            "region_valid": torch.ones(batch, 32, dtype=torch.bool),
        }


@pytest.mark.parametrize("variant", FrozenCXRCLIPHierarchyModel.VARIANTS)
def test_controlled_variant_end_to_end_contract(variant):
    model = FrozenCXRCLIPHierarchyModel(
        checkpoint_path="unused",
        variant=variant,
        feature_extractor=FakeFeatureExtractor(),
    )
    output = model(
        torch.randn(2, 3, 224, 224),
        torch.zeros(2, 32, 4),
        torch.zeros(2, 32, dtype=torch.bool),
    )
    assert output["c1_logits"].shape == (2, 32, 5)
    assert output["c2_logits"].shape == (2, 32, 43)
    assert output["c3_logits"].shape == (2, 10)
    assert output["region_valid"].shape == (2, 32)


def test_c_controls_are_exactly_capacity_matched():
    counts = []
    for variant in ("c", "c_zero", "c_shuffle"):
        model = FrozenCXRCLIPHierarchyModel(
            checkpoint_path="unused",
            variant=variant,
            feature_extractor=FakeFeatureExtractor(),
        )
        counts.append(sum(parameter.numel() for parameter in model.reasoning.parameters()))
    assert len(set(counts)) == 1


def test_invalid_variant_is_rejected():
    with pytest.raises(ValueError, match="variant"):
        FrozenCXRCLIPHierarchyModel(
            checkpoint_path="unused",
            variant="invalid",
            feature_extractor=FakeFeatureExtractor(),
        )
