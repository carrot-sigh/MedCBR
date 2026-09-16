"""Train controlled B/C hierarchy variants with a frozen CXR-CLIP backbone."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.models.cxr_c3_baseline import sha256_file
from src.models.cxr_hierarchy_model import FrozenCXRCLIPHierarchyModel
from src.train_c3_baseline import (
    distributed_runtime,
    make_loaders,
    set_seed,
    unwrap_model,
)
from src.utils.c3_multilabel import (
    masked_bce_with_logits,
    masked_multilabel_metrics,
    tune_f1_thresholds,
)
from src.utils.mimic_hierarchy_dataset import (
    ONTOLOGY_VERSION,
    transform_profile_metadata,
)
from src.utils.hierarchy_screening_subset import load_or_create_screening_subset


LEVEL_FIELDS = {
    "c1": ("c1_logits", "c1_target", "c1_mask", "c1_concepts"),
    "c2": ("c2_logits", "c2_target", "c2_mask", "c2_concepts"),
    "c3": ("c3_logits", "c3_target", "c3_mask", "c3_concepts"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/mimic/cxr_hierarchy_screening.yaml"))
    parser.add_argument("--variant", required=True, choices=FrozenCXRCLIPHierarchyModel.VARIANTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-eval-batches", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_config(args):
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("ontology_version") != ONTOLOGY_VERSION:
        raise ValueError(f"Expected ontology_version={ONTOLOGY_VERSION}")
    config["variant"] = args.variant
    config["output_dir"] = str(
        args.output_dir or Path(config["output_root"]) / args.variant
    )
    overrides = {
        (None, "device"): args.device,
        ("training", "epochs"): args.epochs,
        ("data", "batch_size"): args.batch_size,
        ("data", "num_workers"): args.num_workers,
    }
    for (section, key), value in overrides.items():
        if value is None:
            continue
        if section is None:
            config[key] = value
        else:
            config[section][key] = value
    if args.smoke_test:
        config["training"]["epochs"] = 1
        args.limit_train_batches = args.limit_train_batches or 2
        args.limit_eval_batches = args.limit_eval_batches or 2
    return config


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


def save_history(path, history):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def hierarchy_support(dataset, ontology, indices=None):
    result = {}
    table = dataset.table
    if indices is not None:
        table = table.take(pa.array(indices))
    for level, (_, target_field, mask_field, concept_field) in LEVEL_FIELDS.items():
        concepts = ontology[concept_field]
        positive = np.zeros(len(concepts), dtype=np.int64)
        negative = np.zeros(len(concepts), dtype=np.int64)
        unknown = np.zeros(len(concepts), dtype=np.int64)
        columns = table.select([target_field, mask_field])
        for record_batch in columns.to_batches(max_chunksize=2048):
            target = np.asarray(record_batch.column(0).to_pylist(), dtype=np.int8)
            mask = np.asarray(record_batch.column(1).to_pylist(), dtype=bool)
            target = target.reshape(-1, target.shape[-1])
            mask = mask.reshape(-1, mask.shape[-1])
            positive += (mask & (target == 1)).sum(axis=0)
            negative += (mask & (target == 0)).sum(axis=0)
            unknown += (~mask).sum(axis=0)
        result[level] = [
            {
                "concept": concept,
                "positive": int(positive[index]),
                "negative": int(negative[index]),
                "unknown": int(unknown[index]),
            }
            for index, concept in enumerate(concepts)
        ]
    return result


def subset_train_loader(config, dataset, indices, rank, world_size):
    subset = Subset(dataset, indices.tolist())
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            subset, num_replicas=world_size, rank=rank, shuffle=True,
            seed=int(config["seed"]), drop_last=False,
        )
    workers = int(config["data"]["num_workers"])
    loader = DataLoader(
        subset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    return loader, sampler


def distributed_loss(logits, target, mask, world_size):
    local_loss = masked_bce_with_logits(logits, target, mask)
    local_count = mask.sum().to(dtype=local_loss.dtype)
    backward_loss = local_loss
    if world_size > 1:
        global_count = local_count.detach().clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        backward_loss = local_loss * local_count * world_size / global_count.clamp_min(1)
    return local_loss, backward_loss, local_count


def reduce_loss_totals(numerators, counts, device, world_size):
    packed = torch.tensor(
        [value for level in LEVEL_FIELDS for value in (numerators[level], counts[level])],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    values = packed.cpu().tolist()
    metrics = {}
    for index, level in enumerate(LEVEL_FIELDS):
        numerator, count = values[index * 2:index * 2 + 2]
        metrics[f"{level}_loss"] = numerator / max(count, 1)
        metrics[f"{level}_explicit"] = int(count)
    return metrics


def train_epoch(model, loader, optimizer, scaler, device, config, rank, world_size, max_batches, epoch):
    model.train()
    numerators = {level: 0.0 for level in LEVEL_FIELDS}
    counts = {level: 0 for level in LEVEL_FIELDS}
    amp_enabled = scaler.is_enabled()
    progress = tqdm(loader, desc=f"train {epoch}", disable=rank != 0)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        images = batch["img"].to(device, non_blocking=True)
        boxes = batch["region_bboxes"].to(device, non_blocking=True)
        bbox_mask = batch["region_bbox_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(images, boxes, bbox_mask)
            backward = 0.0
            for level, (logit_field, target_field, mask_field, _) in LEVEL_FIELDS.items():
                target = batch[target_field].to(device, dtype=torch.float32, non_blocking=True)
                mask = batch[mask_field].to(device, dtype=torch.bool, non_blocking=True)
                local_loss, scaled_loss, local_count = distributed_loss(
                    output[logit_field], target, mask, world_size
                )
                backward = backward + float(config["loss_weights"][level]) * scaled_loss
                count = int(local_count.item())
                numerators[level] += float(local_loss.detach()) * count
                counts[level] += count
        scaler.scale(backward).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            float(config["training"]["grad_clip_norm"]),
        )
        scaler.step(optimizer)
        scaler.update()
    return reduce_loss_totals(numerators, counts, device, world_size)


def gather_payload(payload, world_size):
    if world_size == 1:
        return [payload]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, payload)
    return gathered


@torch.no_grad()
def evaluate(model, loader, device, ontology, rank, world_size, max_batches, desc):
    model.eval()
    numerators = {level: 0.0 for level in LEVEL_FIELDS}
    counts = {level: 0 for level in LEVEL_FIELDS}
    arrays = {
        level: {"prob": [], "target": [], "mask": []}
        for level in LEVEL_FIELDS
    }
    for batch_index, batch in enumerate(tqdm(loader, desc=desc, disable=rank != 0)):
        if max_batches is not None and batch_index >= max_batches:
            break
        output = model(
            batch["img"].to(device, non_blocking=True),
            batch["region_bboxes"].to(device, non_blocking=True),
            batch["region_bbox_mask"].to(device, non_blocking=True),
        )
        for level, (logit_field, target_field, mask_field, _) in LEVEL_FIELDS.items():
            target = batch[target_field].to(device, dtype=torch.float32, non_blocking=True)
            mask = batch[mask_field].to(device, dtype=torch.bool, non_blocking=True)
            loss = masked_bce_with_logits(output[logit_field], target, mask)
            count = int(mask.sum().item())
            numerators[level] += float(loss) * count
            counts[level] += count
            arrays[level]["prob"].append(torch.sigmoid(output[logit_field]).float().cpu().numpy())
            arrays[level]["target"].append(target.cpu().numpy())
            arrays[level]["mask"].append(mask.cpu().numpy())

    payload = {"numerators": numerators, "counts": counts, "arrays": {}}
    for level in LEVEL_FIELDS:
        payload["arrays"][level] = {
            key: np.concatenate(value) for key, value in arrays[level].items()
        }
    gathered = gather_payload(payload, world_size)
    losses = {}
    merged = {}
    for level, (_, _, _, concept_field) in LEVEL_FIELDS.items():
        numerator = sum(item["numerators"][level] for item in gathered)
        count = sum(item["counts"][level] for item in gathered)
        losses[f"{level}_loss"] = numerator / max(count, 1)
        level_arrays = {
            key: np.concatenate([item["arrays"][level][key] for item in gathered])
            for key in ("prob", "target", "mask")
        }
        if level != "c3":
            level_arrays = {
                key: value.reshape(-1, value.shape[-1])
                for key, value in level_arrays.items()
            }
        merged[level] = level_arrays
        level_metrics = masked_multilabel_metrics(
            level_arrays["prob"], level_arrays["target"], level_arrays["mask"],
            ontology[concept_field],
        )
        if level != "c3":
            for row in level_metrics["per_class"]:
                row["concept"] = row.pop("diagnosis")
        losses[level] = level_metrics
    return losses, merged


def trainable_state(model):
    base = unwrap_model(model)
    return {
        "region_pooler": base.feature_extractor.region_pooler.state_dict(),
        "reasoning": base.reasoning.state_dict(),
    }


def load_trainable_state(model, state):
    base = unwrap_model(model)
    base.feature_extractor.region_pooler.load_state_dict(state["region_pooler"], strict=True)
    base.reasoning.load_state_dict(state["reasoning"], strict=True)


def build_model(config):
    model = config["model"]
    return FrozenCXRCLIPHierarchyModel(
        checkpoint_path=model["checkpoint"],
        variant=config["variant"],
        spatial_stage=model["spatial_stage"],
        dim=int(model["dim"]),
        predictor_hidden=int(model["predictor_hidden"]),
        semantic_hidden=int(model["semantic_hidden"]),
        c3_hidden=int(model["c3_hidden"]),
        dropout=float(model["dropout"]),
        stop_gradient=bool(model["stop_gradient"]),
        initial_alpha=float(model["initial_alpha"]),
    )


def run(config, args):
    rank, world_size, local_rank, device = distributed_runtime(config["device"])
    is_main = rank == 0
    output_dir = Path(config["output_dir"])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    set_seed(int(config["seed"]) + rank)
    datasets, loaders, train_sampler = make_loaders(config, rank, world_size)
    ontology = datasets["train"].ontology
    subset_indices = None
    subset_report = None
    subset_size = config["data"].get("screening_train_size")
    if subset_size:
        subset_indices, subset_report = load_or_create_screening_subset(
            datasets["train"].table,
            config["data"]["screening_subset_path"],
            int(subset_size),
            int(config["seed"]),
            int(config["data"].get("screening_min_positive_per_concept", 200)),
        )
        loaders["train"], train_sampler = subset_train_loader(
            config, datasets["train"], subset_indices, rank, world_size
        )

    checkpoint_path = Path(config["model"]["checkpoint"])
    digest = sha256_file(checkpoint_path)
    if digest != config["model"]["expected_sha256"]:
        raise RuntimeError("CXR-CLIP checkpoint SHA-256 mismatch")
    preprocessing = transform_profile_metadata(
        config["data"]["transform_profile"], config["data"]["image_size"],
        config["data"]["transform_profile_version"],
    )
    if is_main:
        support = {
            split: hierarchy_support(
                dataset, ontology, subset_indices if split == "train" else None
            )
            for split, dataset in datasets.items()
        }
        save_json(output_dir / "support.json", support)
        if subset_report is not None:
            save_json(output_dir / "train_subset_report.json", subset_report)
        save_json(
            output_dir / "resolved_config.json",
            {**config, "world_size": world_size, "checkpoint_sha256": digest,
             "preprocessing_metadata": preprocessing},
        )

    model = build_model(config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=bool(config["use_amp"]) and device.type == "cuda"
    )
    history = []
    best_score = float("-inf")
    stale = 0
    best_path = output_dir / "best_checkpoint.pt"
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics = train_epoch(
            model, loaders["train"], optimizer, scaler, device, config, rank, world_size,
            args.limit_train_batches, epoch,
        )
        valid_metrics, _ = evaluate(
            model, loaders["valid"], device, ontology, rank, world_size,
            args.limit_eval_batches, f"valid {epoch}",
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            "valid_c1_loss": valid_metrics["c1_loss"],
            "valid_c2_loss": valid_metrics["c2_loss"],
            "valid_c3_loss": valid_metrics["c3_loss"],
            "valid_c3_macro_auroc": valid_metrics["c3"]["macro_auroc"],
            "valid_c3_macro_auprc": valid_metrics["c3"]["macro_auprc"],
        }
        history.append(row)
        score = float(valid_metrics["c3"]["macro_auprc"])
        if is_main:
            logging.info("variant=%s epoch=%d valid_c3_macro_auprc=%.5f", config["variant"], epoch, score)
            save_json(output_dir / "history.json", history)
            save_history(output_dir / "history.csv", history)
        if np.isfinite(score) and score > best_score:
            best_score = score
            stale = 0
            if is_main:
                torch.save(
                    {
                        **trainable_state(model),
                        "epoch": epoch,
                        "valid_c3_macro_auprc": score,
                        "variant": config["variant"],
                        "seed": config["seed"],
                        "config": config,
                        "checkpoint_sha256": digest,
                    },
                    best_path,
                )
        else:
            stale += 1
        if stale >= int(config["training"]["early_stopping_patience"]):
            break

    if world_size > 1:
        dist.barrier()
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    load_trainable_state(model, checkpoint)
    valid_metrics, valid_arrays = evaluate(
        model, loaders["valid"], device, ontology, rank, world_size,
        args.limit_eval_batches, "valid final",
    )
    thresholds = tune_f1_thresholds(
        valid_arrays["c3"]["prob"], valid_arrays["c3"]["target"], valid_arrays["c3"]["mask"]
    )
    valid_metrics["c3"] = masked_multilabel_metrics(
        valid_arrays["c3"]["prob"], valid_arrays["c3"]["target"],
        valid_arrays["c3"]["mask"], ontology["c3_concepts"], thresholds,
    )
    test_metrics, test_arrays = evaluate(
        model, loaders["test"], device, ontology, rank, world_size,
        args.limit_eval_batches, "test final",
    )
    test_metrics["c3"] = masked_multilabel_metrics(
        test_arrays["c3"]["prob"], test_arrays["c3"]["target"],
        test_arrays["c3"]["mask"], ontology["c3_concepts"], thresholds,
    )
    if not is_main:
        dist.destroy_process_group()
        return None
    save_json(output_dir / "thresholds.json", dict(zip(ontology["c3_concepts"], thresholds.tolist())))
    save_json(output_dir / "validation_metrics.json", valid_metrics)
    save_json(output_dir / "test_metrics.json", test_metrics)
    summary = {
        "variant": config["variant"],
        "seed": config["seed"],
        "best_epoch": checkpoint["epoch"],
        "best_valid_c3_macro_auprc": checkpoint["valid_c3_macro_auprc"],
        "validation_c3_macro_auroc": valid_metrics["c3"]["macro_auroc"],
        "validation_c3_macro_auprc": valid_metrics["c3"]["macro_auprc"],
        "test_c3_macro_auroc": test_metrics["c3"]["macro_auroc"],
        "test_c3_macro_auprc": test_metrics["c3"]["macro_auprc"],
    }
    save_json(output_dir / "summary.json", summary)
    if world_size > 1:
        dist.destroy_process_group()
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    summary = run(load_config(args), args)
    if summary is not None:
        print(json.dumps(json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
