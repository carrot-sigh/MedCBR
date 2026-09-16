"""Train the image-only MIMIC-CXR Baseline A against hierarchy V1 C3 labels."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.models.cxr_c3_baseline import C3ImageClassifier, sha256_file
from src.utils.c3_multilabel import (
    masked_bce_with_logits,
    masked_multilabel_metrics,
    tune_f1_thresholds,
)
from src.utils.mimic_hierarchy_dataset import (
    MIMICHierarchyV1Dataset,
    ONTOLOGY_VERSION,
    verify_frozen_hierarchy,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/mimic/c3_baseline_a1.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="Official CXR-CLIP SwinTiny checkpoint")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--freeze-encoder-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-eval-batches", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_config(args):
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("ontology_version") != ONTOLOGY_VERSION:
        raise ValueError(f"Expected ontology_version={ONTOLOGY_VERSION}")
    overrides = {
        ("model", "checkpoint"): args.checkpoint,
        (None, "output_dir"): args.output_dir,
        (None, "device"): args.device,
        ("training", "epochs"): args.epochs,
        ("training", "freeze_encoder_epochs"): args.freeze_encoder_epochs,
        ("data", "batch_size"): args.batch_size,
        ("data", "num_workers"): args.num_workers,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            if isinstance(value, Path):
                value = str(value)
            if section is None:
                config[key] = value
            else:
                config[section][key] = value
    if not config["model"].get("checkpoint"):
        raise ValueError("Set model.checkpoint or pass --checkpoint with an official CXR-CLIP SwinTiny checkpoint")
    if args.smoke_test:
        config["training"]["epochs"] = 1
        args.limit_train_batches = args.limit_train_batches or 2
        args.limit_eval_batches = args.limit_eval_batches or 2
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class DistributedEvalSampler(Sampler):
    """Shard evaluation data without DistributedSampler's padding duplicates."""

    def __init__(self, dataset, rank, world_size):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


def make_loaders(config, rank=0, world_size=1):
    data = config["data"]
    manifest_dir = Path(data["manifest_dir"])
    verify_frozen_hierarchy(manifest_dir)
    common = {
        "manifest_dir": manifest_dir,
        "image_size": int(data["image_size"]),
        "transform_profile": data.get("transform_profile", "cxr_clip"),
    }
    datasets = {
        split: MIMICHierarchyV1Dataset(split=split, **common)
        for split in ("train", "valid", "test")
    }
    workers = int(data["num_workers"])
    loader_args = {
        "batch_size": int(data["batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    samplers = {"train": None, "valid": None, "test": None}
    if world_size > 1:
        samplers["train"] = DistributedSampler(
            datasets["train"], num_replicas=world_size, rank=rank,
            shuffle=True, seed=int(config["seed"]), drop_last=False,
        )
        samplers["valid"] = DistributedEvalSampler(datasets["valid"], rank, world_size)
        samplers["test"] = DistributedEvalSampler(datasets["test"], rank, world_size)
    loaders = {
        split: DataLoader(
            datasets[split],
            shuffle=split == "train" and samplers[split] is None,
            sampler=samplers[split],
            **loader_args,
        )
        for split in ("train", "valid", "test")
    }
    return datasets, loaders, samplers["train"]


def distributed_runtime(requested_device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if requested_device == "cpu":
            raise ValueError("Multi-process Baseline A currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, local_rank, torch.device("cuda", local_rank)
    return rank, world_size, local_rank, resolve_device(requested_device)


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def run_epoch(
    model,
    loader,
    device,
    class_names,
    optimizer=None,
    scaler=None,
    grad_clip_norm=1.0,
    encoder_frozen=False,
    max_batches=None,
    desc="train",
    rank=0,
    world_size=1,
):
    training = optimizer is not None
    model.train(training)
    if training and encoder_frozen:
        unwrap_model(model).image_encoder.eval()
    amp_enabled = scaler is not None and scaler.is_enabled()
    total_loss = 0.0
    total_explicit = 0
    probabilities, targets, masks = [], [], []

    for batch_index, batch in enumerate(tqdm(loader, desc=desc, disable=rank != 0)):
        if max_batches is not None and batch_index >= max_batches:
            break
        # Baseline A deliberately consumes only these three hierarchy fields.
        images = batch["img"].to(device, non_blocking=True)
        target = batch["c3_target"].to(device, dtype=torch.float32, non_blocking=True)
        mask = batch["c3_mask"].to(device, dtype=torch.bool, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), _autocast(device, amp_enabled):
            logits = model(images)
            loss = masked_bce_with_logits(logits, target, mask)
        if training:
            backward_loss = loss
            if world_size > 1:
                local_explicit = mask.sum().to(dtype=loss.dtype)
                global_explicit = local_explicit.detach().clone()
                dist.all_reduce(global_explicit, op=dist.ReduceOp.SUM)
                backward_loss = loss * local_explicit * world_size / global_explicit.clamp_min(1)
            scaler.scale(backward_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        explicit = int(mask.sum().item())
        total_loss += float(loss.detach()) * explicit
        total_explicit += explicit
        probabilities.append(torch.sigmoid(logits.detach()).float().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())

    if not probabilities:
        raise RuntimeError(f"No batches were processed for {desc}")
    payload = (
        np.concatenate(probabilities), np.concatenate(targets), np.concatenate(masks),
        total_loss, total_explicit,
    )
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
        probabilities = np.concatenate([item[0] for item in gathered])
        targets = np.concatenate([item[1] for item in gathered])
        masks = np.concatenate([item[2] for item in gathered])
        total_loss = sum(item[3] for item in gathered)
        total_explicit = sum(item[4] for item in gathered)
    else:
        probabilities, targets, masks = payload[:3]
    metrics = masked_multilabel_metrics(probabilities, targets, masks, class_names)
    metrics["loss"] = total_loss / max(total_explicit, 1)
    metrics["explicit_labels"] = total_explicit
    return metrics, probabilities, targets, masks


def scalar_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "per_class"}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2), encoding="utf-8")


def save_history_csv(path, history):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def label_support(dataset, class_names):
    targets = np.asarray(dataset.table.column("c3_target").to_pylist(), dtype=np.int8)
    masks = np.asarray(dataset.table.column("c3_mask").to_pylist(), dtype=bool)
    rows = []
    for class_index, class_name in enumerate(class_names):
        rows.append(
            {
                "diagnosis": class_name,
                "positive": int((masks[:, class_index] & (targets[:, class_index] == 1)).sum()),
                "negative": int((masks[:, class_index] & (targets[:, class_index] == 0)).sum()),
                "unknown": int((~masks[:, class_index]).sum()),
            }
        )
    return rows


def save_per_class_csv(path, validation_metrics, test_metrics):
    validation = {row["diagnosis"]: row for row in validation_metrics["per_class"]}
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "diagnosis", "threshold", "valid_auroc", "valid_auprc", "valid_f1",
            "test_auroc", "test_auprc", "test_f1", "test_positive", "test_negative",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_metrics["per_class"]:
            valid = validation[row["diagnosis"]]
            writer.writerow(
                {
                    "diagnosis": row["diagnosis"],
                    "threshold": row["threshold"],
                    "valid_auroc": valid["auroc"],
                    "valid_auprc": valid["auprc"],
                    "valid_f1": valid["f1"],
                    "test_auroc": row["auroc"],
                    "test_auprc": row["auprc"],
                    "test_f1": row["f1"],
                    "test_positive": row["positive"],
                    "test_negative": row["negative"],
                }
            )


def save_curves(path, history):
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], label="valid")
    axes[0].set_title("Masked BCE")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_macro_auprc"] for row in history], label="train")
    axes[1].plot(epochs, [row["valid_macro_auprc"] for row in history], label="valid")
    axes[1].set_title("Macro AUPRC")
    axes[1].legend()
    axes[2].plot(epochs, [row["train_macro_auroc"] for row in history], label="train")
    axes[2].plot(epochs, [row["valid_macro_auroc"] for row in history], label="valid")
    axes[2].set_title("Macro AUROC")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train(config, args):
    rank, world_size, local_rank, device = distributed_runtime(config["device"])
    is_main = rank == 0
    output_dir = Path(config["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    set_seed(int(config["seed"]) + rank)
    logging.info("rank=%d/%d using device %s", rank, world_size, device)

    datasets, loaders, train_sampler = make_loaders(config, rank, world_size)
    class_names = datasets["train"].ontology["c3_concepts"]
    if len(class_names) != int(config["model"]["num_classes"]):
        raise ValueError("C3 ontology width and model.num_classes differ")
    if is_main:
        support = {split: label_support(dataset, class_names) for split, dataset in datasets.items()}
        save_json(output_dir / "label_support.json", support)
        for split in ("valid", "test"):
            evaluable = sum(row["positive"] > 0 and row["negative"] > 0 for row in support[split])
            if evaluable < len(class_names):
                logging.warning(
                    "%s has explicit positive and negative labels for only %d/%d C3 classes; "
                    "undefined metrics remain null and unknown labels are not converted to negatives",
                    split, evaluable, len(class_names),
                )

    checkpoint_path = Path(config["model"]["checkpoint"])
    checkpoint_digest = sha256_file(checkpoint_path)
    expected_digest = config["model"].get("expected_sha256")
    if expected_digest and checkpoint_digest != expected_digest:
        raise RuntimeError(
            f"CXR-CLIP checkpoint SHA-256 mismatch: {checkpoint_digest} != {expected_digest}"
        )
    model = C3ImageClassifier(checkpoint_path, num_classes=len(class_names)).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": float(training["encoder_lr"])},
            {"params": model.c3_head.parameters(), "lr": float(training["head_lr"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True,
        )
    amp_enabled = bool(config["use_amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    checkpoint_metadata = {
        "source": "official CXR-CLIP SwinTiny",
        "source_url": config["model"].get("source_url"),
        "path": str(checkpoint_path.resolve()),
        "sha256": checkpoint_digest,
    }
    if is_main:
        save_json(
            output_dir / "resolved_config.json",
            {**config, "checkpoint_metadata": checkpoint_metadata, "world_size": world_size},
        )

    history = []
    best_auprc = float("-inf")
    stale_epochs = 0
    best_path = output_dir / "best_checkpoint.pt"
    freeze_epochs = int(training["freeze_encoder_epochs"])

    for epoch in range(int(training["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        encoder_frozen = epoch < freeze_epochs
        unwrap_model(model).set_encoder_trainable(not encoder_frozen)
        train_metrics, _, _, _ = run_epoch(
            model, loaders["train"], device, class_names, optimizer=optimizer, scaler=scaler,
            grad_clip_norm=float(training["grad_clip_norm"]), encoder_frozen=encoder_frozen,
            max_batches=args.limit_train_batches, desc=f"train {epoch + 1}",
            rank=rank, world_size=world_size,
        )
        valid_metrics, _, _, _ = run_epoch(
            model, loaders["valid"], device, class_names,
            max_batches=args.limit_eval_batches, desc=f"valid {epoch + 1}",
            rank=rank, world_size=world_size,
        )
        row = {
            "epoch": epoch + 1,
            "encoder_frozen": encoder_frozen,
            **{f"train_{key}": value for key, value in scalar_metrics(train_metrics).items()},
            **{f"valid_{key}": value for key, value in scalar_metrics(valid_metrics).items()},
        }
        history.append(row)
        if is_main:
            logging.info("epoch=%d train_loss=%.5f valid_loss=%.5f valid_macro_auprc=%.5f",
                         epoch + 1, row["train_loss"], row["valid_loss"], row["valid_macro_auprc"])
            save_json(output_dir / "history.json", history)
            save_history_csv(output_dir / "history.csv", history)
            save_curves(output_dir / "training_curves.png", history)

        score = float(valid_metrics["macro_auprc"])
        if np.isfinite(score) and score > best_auprc:
            best_auprc = score
            stale_epochs = 0
            if is_main:
                torch.save(
                    {
                        "model": unwrap_model(model).state_dict(),
                        "epoch": epoch + 1,
                        "valid_macro_auprc": score,
                        "ontology_version": ONTOLOGY_VERSION,
                        "c3_concepts": class_names,
                        "checkpoint_metadata": checkpoint_metadata,
                        "config": config,
                    },
                    best_path,
                )
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stopping_patience"]):
            if is_main:
                logging.info("Early stopping after %d stale epochs", stale_epochs)
            break

    if world_size > 1:
        dist.barrier()
    if not best_path.is_file():
        raise RuntimeError("No best checkpoint was saved; validation macro AUPRC was never finite")
    best = torch.load(best_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(best["model"], strict=True)
    unwrap_model(model).set_encoder_trainable(True)

    valid_metrics, valid_prob, valid_target, valid_mask = run_epoch(
        model, loaders["valid"], device, class_names,
        max_batches=args.limit_eval_batches, desc="valid final",
        rank=rank, world_size=world_size,
    )
    thresholds = tune_f1_thresholds(valid_prob, valid_target, valid_mask)
    valid_metrics = masked_multilabel_metrics(
        valid_prob, valid_target, valid_mask, class_names, thresholds
    ) | {"loss": valid_metrics["loss"]}
    test_metrics, test_prob, test_target, test_mask = run_epoch(
        model, loaders["test"], device, class_names,
        max_batches=args.limit_eval_batches, desc="test final",
        rank=rank, world_size=world_size,
    )
    test_metrics = masked_multilabel_metrics(
        test_prob, test_target, test_mask, class_names, thresholds
    ) | {"loss": test_metrics["loss"]}

    if not is_main:
        dist.destroy_process_group()
        return None
    save_json(output_dir / "thresholds.json", dict(zip(class_names, thresholds.tolist())))
    save_json(output_dir / "validation_metrics.json", valid_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)
    save_per_class_csv(output_dir / "per_class_metrics.csv", valid_metrics, test_metrics)
    summary = {
        "best_epoch": best["epoch"],
        "best_valid_macro_auprc": best["valid_macro_auprc"],
        "validation": scalar_metrics(valid_metrics),
        "test": scalar_metrics(test_metrics),
        "artifacts": {
            "checkpoint": str(best_path),
            "training_curves": str(output_dir / "training_curves.png"),
            "history": str(output_dir / "history.csv"),
            "label_support": str(output_dir / "label_support.json"),
            "thresholds": str(output_dir / "thresholds.json"),
            "per_class_metrics": str(output_dir / "per_class_metrics.csv"),
        },
    }
    save_json(output_dir / "summary.json", summary)
    if world_size > 1:
        dist.destroy_process_group()
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args)
    summary = train(config, args)
    if summary is not None:
        print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
