import random

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.mimic_hierarchy_dataset import (
    CXRImageTransform,
    TRANSFORM_PROFILES,
    build_transform,
)


def sample_input():
    pixels = np.arange(320 * 256, dtype=np.uint32).reshape(256, 320) % 256
    image = Image.fromarray(pixels.astype(np.uint8), mode="L").convert("RGB")
    boxes = np.asarray([[20, 30, 180, 220], [0, 0, 0, 0]], dtype=np.float32)
    mask = np.asarray([True, False])
    return image, boxes, mask


def test_cxr_clip_profile_preserves_legacy_transform():
    image, boxes, mask = sample_input()
    legacy = CXRImageTransform(224, "train", "huggingface", True)
    profiled = build_transform("cxr_clip", "train", 224)

    random.seed(17)
    torch.manual_seed(17)
    expected = legacy(image.copy(), boxes.copy(), mask.copy())
    random.seed(17)
    torch.manual_seed(17)
    actual = profiled(image.copy(), boxes.copy(), mask.copy())

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize("profile", sorted(TRANSFORM_PROFILES))
def test_profiles_return_uniform_image_and_bbox_contract(profile):
    image, boxes, mask = sample_input()
    tensor, transformed_boxes, transformed_mask = build_transform(
        profile, "valid", 224
    )(image, boxes, mask)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert transformed_boxes.shape == (2, 4)
    assert transformed_mask.dtype == torch.bool
    active = transformed_boxes[transformed_mask]
    assert torch.all(active[:, :2] >= 0)
    assert torch.all(active[:, 2:] <= 224)


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown transform profile"):
        build_transform("unknown", "train", 224)
