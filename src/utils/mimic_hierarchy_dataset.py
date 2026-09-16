"""MIMIC-CXR input pipeline for the frozen Chest ImaGenome hierarchy V1."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


ONTOLOGY_VERSION = "hierarchy_v1"
SPLIT_FILES = {"train": "train.parquet", "valid": "valid.parquet", "test": "test.parquet"}


def verify_frozen_hierarchy(manifest_dir: str | Path) -> None:
    """Verify all frozen manifest files against ``hierarchy_v1.sha256``."""
    manifest_dir = Path(manifest_dir).resolve()
    hierarchy_dir = manifest_dir.parent
    hash_file = hierarchy_dir / "hierarchy_v1.sha256"
    if not hash_file.is_file():
        raise FileNotFoundError(f"Missing frozen hash manifest: {hash_file}")
    for line in hash_file.read_text(encoding="utf-8").splitlines():
        expected, relative_path = line.split(maxsplit=1)
        path = hierarchy_dir / relative_path
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Frozen hierarchy checksum mismatch: {path}")


@dataclass(frozen=True)
class TransformProfile:
    """Backbone-specific image preprocessing parameters."""

    version: int
    normalize: str
    clahe: bool
    color_jitter: bool
    interpolation: InterpolationMode
    train_crop_scale: tuple[float, float]


TRANSFORM_PROFILES = {
    "cxr_clip": TransformProfile(
        version=1, normalize="huggingface", clahe=True, color_jitter=True,
        interpolation=InterpolationMode.BILINEAR, train_crop_scale=(0.8, 1.1),
    ),
    "vit": TransformProfile(
        version=1, normalize="imagenet", clahe=False, color_jitter=False,
        interpolation=InterpolationMode.BICUBIC, train_crop_scale=(0.8, 1.0),
    ),
    "dinov2": TransformProfile(
        version=1, normalize="imagenet", clahe=False, color_jitter=False,
        interpolation=InterpolationMode.BICUBIC, train_crop_scale=(0.8, 1.0),
    ),
}


class CXRImageTransform:
    """Image preprocessing with synchronized bbox geometry."""

    NORMALIZATION = {
        "huggingface": ([0.5] * 3, [0.5] * 3),
        "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }

    def __init__(
        self,
        image_size=224,
        split="train",
        normalize="huggingface",
        clahe=True,
        color_jitter=True,
        interpolation=InterpolationMode.BILINEAR,
        train_crop_scale=(0.8, 1.1),
    ):
        if split not in SPLIT_FILES:
            raise ValueError(f"Unsupported split: {split}")
        if normalize not in self.NORMALIZATION:
            raise ValueError(f"Unsupported normalization: {normalize}")
        self.image_size = int(image_size)
        self.split = split
        self.clahe = clahe
        self.apply_color_jitter = color_jitter
        self.color_jitter = ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1)
        self.interpolation = interpolation
        self.train_crop_scale = tuple(train_crop_scale)
        mean, std = self.NORMALIZATION[normalize]
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def _clahe(self, image: Image.Image) -> Image.Image:
        # Albumentations CLAHE defaults to p=0.5 in CXR-CLIP's train preset.
        if not self.clahe or random.random() >= 0.5:
            return image
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("CLAHE preprocessing requires opencv-python-headless") from exc
        # Albumentations 1.3.1 samples a scalar clip_limit=4.0 from [1.0, 4.0].
        clip_limit = random.uniform(1.0, 4.0)
        lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    @staticmethod
    def _transform_bboxes(bboxes, mask, source_size, crop, output_size):
        source_width, source_height = source_size
        left, top, crop_width, crop_height = crop
        output_width, output_height = output_size
        result = bboxes.astype(np.float32, copy=True)
        result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, source_width)
        result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, source_height)
        result[:, [0, 2]] = (result[:, [0, 2]] - left) * output_width / crop_width
        result[:, [1, 3]] = (result[:, [1, 3]] - top) * output_height / crop_height
        result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, output_width)
        result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, output_height)
        valid = mask & (result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])
        result[~valid] = 0
        return result, valid

    def __call__(self, image, bboxes, bbox_mask):
        source_width, source_height = image.size
        if self.split == "train":
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=self.train_crop_scale, ratio=(3 / 4, 4 / 3)
            )
            image = TF.resized_crop(
                image, top, left, height, width, [self.image_size, self.image_size],
                interpolation=self.interpolation, antialias=True,
            )
            bboxes, bbox_mask = self._transform_bboxes(
                bboxes, bbox_mask, (source_width, source_height),
                (left, top, width, height), (self.image_size, self.image_size),
            )
            image = self._clahe(image)
            if self.apply_color_jitter:
                image = self.color_jitter(image)
        else:
            image = TF.resize(
                image, self.image_size, interpolation=self.interpolation, antialias=True
            )
            resized_width, resized_height = image.size
            crop_left = int(round((resized_width - self.image_size) / 2.0))
            crop_top = int(round((resized_height - self.image_size) / 2.0))
            image = TF.center_crop(image, [self.image_size, self.image_size])
            bboxes, bbox_mask = self._transform_bboxes(
                bboxes, bbox_mask, (source_width, source_height),
                (
                    crop_left * source_width / resized_width,
                    crop_top * source_height / resized_height,
                    self.image_size * source_width / resized_width,
                    self.image_size * source_height / resized_height,
                ),
                (self.image_size, self.image_size),
            )
            # CXR-CLIP applies CLAHE only to its train split.
            # Validation/test is Resize -> CenterCrop.
        image = TF.pil_to_tensor(image).float().div_(255.0)
        image = (image - self.mean) / self.std
        return image, torch.from_numpy(bboxes), torch.from_numpy(bbox_mask)


def transform_profile_metadata(profile="cxr_clip", image_size=224, expected_version=None):
    """Resolve a profile into JSON-serializable, versioned experiment metadata."""
    try:
        spec = TRANSFORM_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(TRANSFORM_PROFILES))
        raise ValueError(f"Unknown transform profile {profile!r}; choose one of: {choices}") from exc
    if expected_version is not None and int(expected_version) != spec.version:
        raise ValueError(
            f"Transform profile {profile!r} version mismatch: "
            f"config={expected_version}, registry={spec.version}"
        )
    mean, std = CXRImageTransform.NORMALIZATION[spec.normalize]
    return {
        "profile": profile,
        "version": spec.version,
        "image_size": int(image_size),
        "mean": list(mean),
        "std": list(std),
        "interpolation": spec.interpolation.value,
        "train_random_resized_crop_scale": list(spec.train_crop_scale),
        "train_random_resized_crop_ratio": [3 / 4, 4 / 3],
        "eval_resize_short_edge": int(image_size),
        "eval_center_crop": int(image_size),
        "crop_pct": 1.0,
        "clahe": spec.clahe,
        "color_jitter": spec.color_jitter,
    }


def build_transform(profile="cxr_clip", split="train", image_size=224, expected_version=None):
    """Build a bbox-aware transform for a named backbone profile."""
    transform_profile_metadata(profile, image_size, expected_version)
    spec = TRANSFORM_PROFILES[profile]
    return CXRImageTransform(
        image_size=image_size,
        split=split,
        normalize=spec.normalize,
        clahe=spec.clahe,
        color_jitter=spec.color_jitter,
        interpolation=spec.interpolation,
        train_crop_scale=spec.train_crop_scale,
    )


class MIMICHierarchyV1Dataset(Dataset):
    """One MIMIC-CXR JPG with frozen region-level hierarchy supervision."""

    TENSOR_FIELDS = (
        "c1_target", "c1_mask", "c2_target", "c2_mask",
        "severity_target", "severity_mask", "c3_target", "c3_mask",
        "c1_quality_weight", "region_active_c1",
    )

    def __init__(
        self,
        manifest_dir,
        split,
        image_size=224,
        transform_profile="cxr_clip",
        transform_profile_version=None,
        transform: Callable | None = None,
        normalize=None,
        clahe=None,
        verify_hashes=False,
    ):
        if split not in SPLIT_FILES:
            raise ValueError(f"Unsupported split: {split}")
        self.manifest_dir = Path(manifest_dir)
        if verify_hashes:
            verify_frozen_hierarchy(self.manifest_dir)
        self.ontology = json.loads((self.manifest_dir / "ontology_v1.json").read_text(encoding="utf-8"))
        self.ontology_version = ONTOLOGY_VERSION
        self.split = split
        self.table = pq.read_table(self.manifest_dir / SPLIT_FILES[split], memory_map=True)
        self.transform_profile = transform_profile
        self.transform_profile_version = transform_profile_version
        if transform is not None and (normalize is not None or clahe is not None):
            raise ValueError("normalize/clahe overrides cannot be combined with a custom transform")
        if transform is not None:
            self.transform = transform
        elif normalize is not None or clahe is not None:
            # Preserve compatibility for existing CXR-CLIP callers.
            if transform_profile != "cxr_clip":
                raise ValueError("normalize/clahe overrides are only supported for cxr_clip")
            self.transform = CXRImageTransform(
                image_size, split, normalize or "huggingface",
                True if clahe is None else bool(clahe),
            )
        else:
            self.transform = build_transform(
                transform_profile, split, image_size, transform_profile_version
            )

    def __len__(self):
        return self.table.num_rows

    def _value(self, field, index) -> Any:
        return self.table.column(field)[index].as_py()

    def __getitem__(self, index):
        image_path = self._value("image_path", index)
        bboxes = np.asarray(self._value("region_bboxes", index), dtype=np.float32)
        bbox_mask = np.asarray(self._value("region_bbox_mask", index), dtype=np.bool_)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        image, bboxes, bbox_mask = self.transform(image, bboxes, bbox_mask)
        sample = {
            "img": image, "img_name": image_path,
            "dicom_id": self._value("dicom_id", index),
            "study_id": self._value("study_id", index),
            "subject_id": self._value("subject_id", index),
            "ontology_version": self.ontology_version,
            "region_bboxes": bboxes, "region_bbox_mask": bbox_mask,
        }
        for field in self.TENSOR_FIELDS:
            value = np.asarray(self._value(field, index))
            if field.endswith("_mask") or field == "region_active_c1":
                sample[field] = torch.from_numpy(value.astype(np.bool_))
            elif field == "c1_quality_weight":
                sample[field] = torch.from_numpy(value.astype(np.float32))
            else:
                sample[field] = torch.from_numpy(value.astype(np.int8))
        return sample


def make_mimic_hierarchy_loaders(config):
    """Build MedCBR loaders from the frozen hierarchy manifests."""
    manifest_dir = Path(config.data.manifest_dir)
    verify_frozen_hierarchy(manifest_dir)
    common = {
        "manifest_dir": manifest_dir, "image_size": int(config.data.image_size),
        "transform_profile": str(getattr(config.data, "transform_profile", "cxr_clip")),
        "transform_profile_version": getattr(config.data, "transform_profile_version", None),
    }
    datasets = {split: MIMICHierarchyV1Dataset(split=split, **common) for split in SPLIT_FILES}
    workers = int(config.data.num_workers)
    common_loader = {
        "batch_size": int(config.data.batch_size), "num_workers": workers,
        "pin_memory": bool(config.data.pin_memory), "persistent_workers": workers > 0,
    }
    return (
        DataLoader(datasets["train"], shuffle=True, **common_loader),
        DataLoader(datasets["valid"], shuffle=False, **common_loader),
        DataLoader(datasets["test"], shuffle=False, **common_loader),
    )
