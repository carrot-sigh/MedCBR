#!/usr/bin/env python3
"""Validate the frozen MIMIC hierarchy input pipeline against CXR-CLIP."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.mimic_hierarchy_dataset import (  # noqa: E402
    CXRImageTransform,
    MIMICHierarchyV1Dataset,
    SPLIT_FILES,
    verify_frozen_hierarchy,
)


LABEL_FIELDS = ("c1_target", "c1_mask", "c2_target", "c2_mask", "c3_target", "c3_mask")
MEAN = torch.tensor([0.5] * 3).view(3, 1, 1)
STD = torch.tensor([0.5] * 3).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=REPO_ROOT / "workspaces/chest_imagenome_hierarchy_v1/manifests",
    )
    parser.add_argument(
        "--cxr-clip-root", type=Path, default=REPO_ROOT.parent / "cxr-clip"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "workspaces/chest_imagenome_hierarchy_v1/dataset_validation",
    )
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--golden-count", type=int, default=20)
    parser.add_argument("--visual-count", type=int, default=50)
    parser.add_argument("--bbox-count", type=int, default=100)
    parser.add_argument("--label-count", type=int, default=100)
    parser.add_argument("--overfit-size", type=int, default=128)
    parser.add_argument("--overfit-epochs", type=int, default=80)
    parser.add_argument("--skip-overfit", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_to_pil(tensor):
    array = ((tensor.cpu() * STD + MEAN).clamp(0, 1) * 255).byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def make_montage(images, output_path, columns=5, label_height=22):
    if not images:
        return
    width, height = images[0][0].size
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        canvas.paste(image, (x, y))
        draw.text((x + 3, y + height + 3), label[:35], fill="black")
    canvas.save(output_path)


def draw_bboxes(image, bboxes, mask, regions):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    palette = ("#00ff65", "#ff3355", "#00b7ff", "#ffd400", "#dd66ff")
    for region_index in torch.where(mask)[0].tolist():
        x1, y1, x2, y2 = bboxes[region_index].tolist()
        color = palette[region_index % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((x1 + 2, y1 + 2), regions[region_index], fill=color, stroke_width=1, stroke_fill="black")
    return image


def load_reference_functions(cxr_clip_root):
    module_path = cxr_clip_root / "cxrclip/data/data_utils.py"
    spec = importlib.util.spec_from_file_location("cxrclip_reference_data_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_transform, module.transform_image


def golden_sample_audit(datasets, cxr_clip_root, count, seed):
    load_transform, transform_image = load_reference_functions(cxr_clip_root)
    config = {
        split: {"Resize": {"size": 224}, "CenterCrop": {"size": 224}}
        for split in ("valid", "test")
    }
    rng = random.Random(seed)
    per_split = {"valid": count // 2, "test": count - count // 2}
    rows = []
    for split, sample_count in per_split.items():
        reference_transform = load_transform(split=split, transform_config=config)
        for index in rng.sample(range(len(datasets[split])), sample_count):
            sample = datasets[split][index]
            with Image.open(sample["img_name"]) as handle:
                reference = transform_image(reference_transform, handle.convert("RGB"), normalize="huggingface")
            current = sample["img"]
            difference = (reference - current).abs()
            rows.append(
                {
                    "split": split,
                    "index": index,
                    "dicom_id": sample["dicom_id"],
                    "reference_shape": list(reference.shape),
                    "current_shape": list(current.shape),
                    "mean_abs_difference": abs(reference.mean().item() - current.mean().item()),
                    "std_abs_difference": abs(reference.std().item() - current.std().item()),
                    "max_abs_pixel_difference": difference.max().item(),
                }
            )
    train_config = {
        "train": {
            "RandomResizedCrop": {"size": 224, "scale": [0.8, 1.1]},
            "CLAHE": {"clip_limit": 4.0},
            "ColorJitter": {"brightness": 0.1, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
            "CenterCrop": {"size": 224},
        }
    }
    reference_train_transform = load_transform(split="train", transform_config=train_config)
    current_train_transform = CXRImageTransform(224, "train", "huggingface", True)
    train_rows = []
    for draw_index, index in enumerate(rng.sample(range(len(datasets["train"])), 10)):
        image_path = datasets["train"]._value("image_path", index)
        bboxes = np.asarray(datasets["train"]._value("region_bboxes", index), dtype=np.float32)
        bbox_mask = np.asarray(datasets["train"]._value("region_bbox_mask", index), dtype=np.bool_)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        draw_seed = seed + 1000 + draw_index
        set_seed(draw_seed)
        reference = transform_image(reference_train_transform, image.copy(), normalize="huggingface")
        set_seed(draw_seed)
        current, _, _ = current_train_transform(image.copy(), bboxes, bbox_mask)
        train_rows.append(
            {
                "index": index,
                "dicom_id": datasets["train"]._value("dicom_id", index),
                "seed": draw_seed,
                "max_abs_pixel_difference": (reference - current).abs().max().item(),
            }
        )
    return {
        "sample_count": len(rows),
        "reference": "cxrclip.data.data_utils.load_transform/transform_image",
        "operation_order": "load RGB -> Resize -> CenterCrop -> ToTensor -> Normalize",
        "shape_passed": all(row["reference_shape"] == row["current_shape"] == [3, 224, 224] for row in rows),
        "max_mean_abs_difference": max(row["mean_abs_difference"] for row in rows),
        "max_std_abs_difference": max(row["std_abs_difference"] for row in rows),
        "max_abs_pixel_difference": max(row["max_abs_pixel_difference"] for row in rows),
        "allclose_atol_1e-7": all(row["max_abs_pixel_difference"] <= 1e-7 for row in rows),
        "train_fixed_seed": {
            "sample_count": len(train_rows),
            "max_abs_pixel_difference": max(row["max_abs_pixel_difference"] for row in train_rows),
            "allclose_atol_1e-7": all(row["max_abs_pixel_difference"] <= 1e-7 for row in train_rows),
            "samples": train_rows,
        },
        "samples": rows,
    }


def image_visual_audit(datasets, output_dir, count, seed):
    rng = random.Random(seed)
    counts = {"train": count // 2, "valid": count - count // 2}
    outputs = []
    sampled = []
    for split, sample_count in counts.items():
        montage_rows = []
        for index in rng.sample(range(len(datasets[split])), sample_count):
            sample = datasets[split][index]
            image = tensor_to_pil(sample["img"])
            montage_rows.append((image, f"{split} {sample['dicom_id'][:18]}"))
            sampled.append(sample)
        output_path = output_dir / f"{split}_images_montage.png"
        make_montage(montage_rows, output_path)
        outputs.append(str(output_path))
    pixels = torch.stack([sample["img"] for sample in sampled])
    return {
        "sample_count": len(sampled),
        "montages": outputs,
        "nan_images": int(torch.isnan(pixels).flatten(1).any(1).sum()),
        "image_mean": pixels.mean(dim=(0, 2, 3)).tolist(),
        "image_std": pixels.std(dim=(0, 2, 3)).tolist(),
        "manual_review_required": True,
        "review_items": [
            "orientation", "left_right_mirroring", "lung_crop", "clahe_strength",
            "ap_pa_appearance", "grayscale_polarity",
        ],
    }


def bbox_audit(dataset, output_dir, count, regions, seed):
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)
    overlay_dir = output_dir / "bbox_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    source_active = 0
    transformed_active = 0
    invalid = 0
    survival_rates = []
    montage_rows = []
    montage_paths = []
    for audit_index, index in enumerate(indices):
        source_mask = np.asarray(dataset._value("region_bbox_mask", index), dtype=np.bool_)
        sample = dataset[index]
        boxes = sample["region_bboxes"]
        mask = sample["region_bbox_mask"]
        active = boxes[mask]
        valid = (
            (active[:, 0] >= 0) & (active[:, 1] >= 0)
            & (active[:, 0] < active[:, 2]) & (active[:, 1] < active[:, 3])
            & (active[:, 2] <= 224) & (active[:, 3] <= 224)
        )
        invalid += int((~valid).sum())
        source_count = int(source_mask.sum())
        transformed_count = int(mask.sum())
        source_active += source_count
        transformed_active += transformed_count
        survival_rates.append(transformed_count / source_count if source_count else 1.0)
        overlay = draw_bboxes(tensor_to_pil(sample["img"]), boxes, mask, regions)
        overlay_path = overlay_dir / f"{audit_index:03d}_{sample['dicom_id']}.png"
        overlay.save(overlay_path)
        montage_rows.append((overlay, f"{survival_rates[-1]:.2f} {sample['dicom_id'][:14]}"))
        if len(montage_rows) == 25 or audit_index == len(indices) - 1:
            montage_path = output_dir / f"bbox_montage_{len(montage_paths) + 1:02d}.png"
            make_montage(montage_rows, montage_path)
            montage_paths.append(str(montage_path))
            montage_rows = []
    return {
        "sample_count": count,
        "invalid_bbox_count": invalid,
        "source_bbox_count": source_active,
        "transformed_bbox_count": transformed_active,
        "bbox_survival_rate": transformed_active / source_active,
        "per_image_survival_p50": float(np.percentile(survival_rates, 50)),
        "per_image_survival_p90": float(np.percentile(survival_rates, 90)),
        "per_image_survival_min": min(survival_rates),
        "images_below_80_percent_survival": sum(rate < 0.8 for rate in survival_rates),
        "overlay_directory": str(overlay_dir),
        "montages": montage_paths,
        "manual_review_required": True,
    }


def label_roundtrip_audit(dataset, count, seed):
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), count)
    expected = {}
    for index in indices:
        dicom_id = dataset._value("dicom_id", index)
        expected[dicom_id] = {
            field: torch.as_tensor(np.asarray(dataset._value(field, index))) for field in LABEL_FIELDS
        }
    loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=True, num_workers=0)
    mismatches = []
    checked = 0
    observed_order = []
    for batch in loader:
        for row_index, dicom_id in enumerate(batch["dicom_id"]):
            observed_order.append(dicom_id)
            checked += 1
            for field in LABEL_FIELDS:
                actual = batch[field][row_index].cpu()
                target = expected[dicom_id][field].to(actual.dtype)
                if not torch.equal(actual, target):
                    mismatches.append({"dicom_id": dicom_id, "field": field})
    return {
        "sample_count": checked,
        "shuffle_enabled": True,
        "shuffle_changed_order": observed_order != [dataset._value("dicom_id", index) for index in indices],
        "mismatch_count": len(mismatches),
        "passed": not mismatches,
        "mismatches": mismatches[:20],
    }


def determinism_audit(datasets, seed):
    set_seed(seed)
    eval_results = {}
    for split in ("valid", "test"):
        index = min(7, len(datasets[split]) - 1)
        first = datasets[split][index]["img"]
        second = datasets[split][index]["img"]
        eval_results[split] = {
            "equal": torch.equal(first, second),
            "max_abs_difference": (first - second).abs().max().item(),
        }
    index = min(7, len(datasets["train"]) - 1)
    train_first = datasets["train"][index]["img"]
    train_second = datasets["train"][index]["img"]
    return {
        "eval": eval_results,
        "eval_deterministic": all(result["equal"] for result in eval_results.values()),
        "train_outputs_differ": not torch.equal(train_first, train_second),
        "train_max_abs_difference": (train_first - train_second).abs().max().item(),
    }


def split_and_path_audit(manifest_dir):
    identity = {}
    missing = 0
    for split, filename in SPLIT_FILES.items():
        table = pq.read_table(
            manifest_dir / filename,
            columns=["dicom_id", "study_id", "subject_id", "image_path"],
            memory_map=True,
        )
        identity[split] = {
            field: set(table.column(field).to_pylist()) for field in ("dicom_id", "study_id", "subject_id")
        }
        missing += sum(not Path(path).is_file() for path in table.column("image_path").to_pylist())
    overlap = {}
    for field in ("dicom_id", "study_id", "subject_id"):
        overlap[field] = sum(
            len(identity[left][field] & identity[right][field])
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test"))
        )
    return {"missing_images": missing, "split_overlap": overlap, "split_overlap_total": sum(overlap.values())}


class FixedC3Dataset(Dataset):
    def __init__(self, hierarchy_dataset, indices):
        transform = CXRImageTransform(image_size=224, split="valid", normalize="huggingface", clahe=False)
        self.rows = []
        for index in indices:
            image_path = hierarchy_dataset._value("image_path", index)
            boxes = np.asarray(hierarchy_dataset._value("region_bboxes", index), dtype=np.float32)
            box_mask = np.asarray(hierarchy_dataset._value("region_bbox_mask", index), dtype=np.bool_)
            with Image.open(image_path) as handle:
                image, _, _ = transform(handle.convert("RGB"), boxes, box_mask)
            target = torch.tensor(hierarchy_dataset._value("c3_target", index), dtype=torch.float32)
            mask = torch.tensor(hierarchy_dataset._value("c3_mask", index), dtype=torch.bool)
            self.rows.append((image, target, mask))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TinyCXRBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        channels = (3, 32, 64, 128, 256)
        layers = [nn.AvgPool2d(2)]
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers.extend(
                [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU()]
            )
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 10)

    def forward(self, image):
        return self.head(self.features(image))


def masked_bce(logits, target, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, target.clamp_min(0), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def c3_metrics(model, loader):
    model.eval()
    logits, targets, masks = [], [], []
    total_loss = 0.0
    batches = 0
    for image, target, mask in loader:
        prediction = model(image)
        total_loss += masked_bce(prediction, target, mask).item()
        batches += 1
        logits.append(prediction.sigmoid())
        targets.append(target)
        masks.append(mask)
    probabilities = torch.cat(logits).numpy()
    targets = torch.cat(targets).numpy()
    masks = torch.cat(masks).numpy().astype(bool)
    aurocs, auprcs = [], []
    for concept in range(targets.shape[1]):
        concept_target = targets[masks[:, concept], concept]
        concept_score = probabilities[masks[:, concept], concept]
        if len(np.unique(concept_target)) == 2:
            aurocs.append(roc_auc_score(concept_target, concept_score))
            auprcs.append(average_precision_score(concept_target, concept_score))
    return {
        "loss": total_loss / batches,
        "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "evaluable_concepts": len(aurocs),
    }


def tiny_overfit_audit(train_dataset, size, epochs, seed, output_dir):
    set_seed(seed)
    rng = random.Random(seed)
    targets = np.asarray(train_dataset.table.column("c3_target").to_pylist(), dtype=np.int8)
    masks = np.asarray(train_dataset.table.column("c3_mask").to_pylist(), dtype=np.bool_)
    selected = set()
    for concept in range(targets.shape[1]):
        for polarity in (0, 1):
            candidates = np.flatnonzero(masks[:, concept] & (targets[:, concept] == polarity)).tolist()
            selected.update(rng.sample(candidates, min(6, len(candidates))))
    if len(selected) > size:
        raise ValueError(f"Stratified C3 seed set ({len(selected)}) exceeds overfit size ({size})")
    remaining = list(set(range(len(train_dataset))) - selected)
    selected.update(rng.sample(remaining, size - len(selected)))
    indices = sorted(selected)
    selected_targets = targets[indices]
    selected_masks = masks[indices]
    explicit_counts = {
        str(concept): {
            "negative": int((selected_masks[:, concept] & (selected_targets[:, concept] == 0)).sum()),
            "positive": int((selected_masks[:, concept] & (selected_targets[:, concept] == 1)).sum()),
        }
        for concept in range(targets.shape[1])
    }
    dataset = FixedC3Dataset(train_dataset, indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    model = TinyCXRBackbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    history = [{"epoch": 0, **c3_metrics(model, eval_loader)}]
    for epoch in range(1, epochs + 1):
        model.train()
        for image, target, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(image), target, mask)
            loss.backward()
            optimizer.step()
        if epoch <= 5 or epoch % 5 == 0 or epoch == epochs:
            history.append({"epoch": epoch, **c3_metrics(model, eval_loader)})
    history_path = output_dir / "tiny_overfit_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    initial, final = history[0], history[-1]
    return {
        "sample_count": size,
        "epochs": epochs,
        "device": "cpu",
        "model": "TinyCXRBackbone + Linear(10)",
        "input_transform": "deterministic CXR-CLIP eval transform",
        "sampling": "stratified with up to 6 explicit positives and 6 explicit negatives per C3 concept",
        "explicit_counts_by_concept_index": explicit_counts,
        "initial": initial,
        "final": final,
        "loss_reduction_fraction": (initial["loss"] - final["loss"]) / initial["loss"],
        "passed": final["loss"] < initial["loss"] * 0.35 and (final["macro_auroc"] or 0) > 0.9,
        "history": str(history_path),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_frozen_hierarchy(args.manifest_dir)
    set_seed(args.seed)
    datasets = {
        split: MIMICHierarchyV1Dataset(
            args.manifest_dir, split, image_size=224, normalize="huggingface", clahe=True
        )
        for split in SPLIT_FILES
    }
    ontology = datasets["train"].ontology
    report = {
        "ontology_version": "hierarchy_v1",
        "seed": args.seed,
        "num_train": len(datasets["train"]),
        "num_valid": len(datasets["valid"]),
        "num_test": len(datasets["test"]),
        "image_shape": [3, 224, 224],
        "preprocessing": {
            "train": "load RGB -> RandomResizedCrop(bilinear) -> CLAHE(p=0.5) -> ColorJitter -> CenterCrop(no-op at 224) -> ToTensor -> Normalize",
            "valid_test": "load RGB -> Resize(bilinear) -> CenterCrop -> ToTensor -> Normalize",
        },
    }
    report["golden_sample"] = golden_sample_audit(
        datasets, args.cxr_clip_root, args.golden_count, args.seed + 1
    )
    report["image_visual_audit"] = image_visual_audit(
        datasets, args.output_dir, args.visual_count, args.seed + 2
    )
    report["bbox_audit"] = bbox_audit(
        datasets["train"], args.output_dir, args.bbox_count, ontology["regions"], args.seed + 3
    )
    report["label_roundtrip"] = label_roundtrip_audit(
        datasets["train"], args.label_count, args.seed + 4
    )
    report["determinism"] = determinism_audit(datasets, args.seed + 5)
    report.update(split_and_path_audit(args.manifest_dir))
    if not args.skip_overfit:
        report["tiny_overfit"] = tiny_overfit_audit(
            datasets["train"], args.overfit_size, args.overfit_epochs, args.seed + 6, args.output_dir
        )
    report["nan_images"] = report["image_visual_audit"]["nan_images"]
    report["invalid_bbox_count"] = report["bbox_audit"]["invalid_bbox_count"]
    report["eval_deterministic"] = report["determinism"]["eval_deterministic"]
    report["label_roundtrip_passed"] = report["label_roundtrip"]["passed"]
    report["bbox_survival_rate"] = report["bbox_audit"]["bbox_survival_rate"]
    report["image_mean"] = report["image_visual_audit"]["image_mean"]
    report["image_std"] = report["image_visual_audit"]["image_std"]
    report_path = args.output_dir / "dataset_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
