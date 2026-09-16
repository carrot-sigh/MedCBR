"""Image-only C3 classifier backed by an official CXR-CLIP Swin encoder."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import SwinConfig, SwinModel


class CXRClipSwinEncoder(nn.Module):
    """The image branch used by the official CXR-CLIP SwinTiny checkpoint."""

    STATE_PREFIXES = (
        "image_encoder.image_encoder.",
        "module.image_encoder.image_encoder.",
        "model.image_encoder.image_encoder.",
    )

    def __init__(self, checkpoint_path: str | Path):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Baseline A1 requires an official CXR-CLIP checkpoint; "
                f"not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self._validate_checkpoint_config(checkpoint)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
        if not isinstance(state_dict, dict):
            raise ValueError("CXR-CLIP checkpoint has no model/state_dict mapping")

        encoder_state = self._extract_encoder_state(state_dict)
        self.encoder = SwinModel(SwinConfig())
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.out_dim = int(self.encoder.config.hidden_size)
        del checkpoint, state_dict, encoder_state

    @classmethod
    def _extract_encoder_state(cls, state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        for prefix in cls.STATE_PREFIXES:
            encoder_state = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if encoder_state:
                return encoder_state
        prefixes = sorted({key.split(".", 1)[0] for key in state_dict})
        raise ValueError(
            "Checkpoint does not contain CXR-CLIP image encoder weights. "
            f"Top-level key prefixes: {prefixes[:20]}"
        )

    @staticmethod
    def _validate_checkpoint_config(checkpoint: dict[str, Any]) -> None:
        config = checkpoint.get("config", {})
        model_config = config.get("model", {}) if isinstance(config, Mapping) else {}
        image_config = model_config.get("image_encoder", {}) if isinstance(model_config, Mapping) else {}
        name = str(image_config.get("name", ""))
        model_type = str(image_config.get("model_type", ""))
        if name and "swin-tiny" not in name.lower():
            raise ValueError(f"Expected CXR-CLIP SwinTiny checkpoint, got image encoder {name!r}")
        if model_type and model_type.lower() != "swin":
            raise ValueError(f"Expected CXR-CLIP Swin encoder, got model_type {model_type!r}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(pixel_values=images).last_hidden_state
        # Match CXR-CLIP's encode_image implementation exactly.
        return tokens[:, 0]


class C3ImageClassifier(nn.Module):
    """CXR-CLIP image encoder followed by a 10-label diagnosis head."""

    def __init__(self, checkpoint_path: str | Path, num_classes: int = 10):
        super().__init__()
        self.image_encoder = CXRClipSwinEncoder(checkpoint_path)
        self.c3_head = nn.Linear(self.image_encoder.out_dim, num_classes)

    def set_encoder_trainable(self, trainable: bool) -> None:
        self.image_encoder.requires_grad_(trainable)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.c3_head(self.image_encoder(images))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
