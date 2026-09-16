"""Deterministic coverage-aware train subsets for hierarchy screening."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


LEVELS = (
    ("c1", "c1_target", "c1_mask"),
    ("c2", "c2_target", "c2_mask"),
    ("c3", "c3_target", "c3_mask"),
)


def _image_positive_matrix(table, target_field, mask_field):
    result = None
    offset = 0
    for batch in table.select([target_field, mask_field]).to_batches(max_chunksize=1024):
        target = np.asarray(batch.column(0).to_pylist(), dtype=np.int8)
        mask = np.asarray(batch.column(1).to_pylist(), dtype=bool)
        positive = mask & (target == 1)
        if positive.ndim == 3:
            positive = positive.any(axis=1)
        if result is None:
            result = np.zeros((table.num_rows, positive.shape[-1]), dtype=bool)
        result[offset:offset + len(positive)] = positive
        offset += len(positive)
    return result


def _coverage(positive_matrices, indices):
    indices = np.asarray(indices, dtype=np.int64)
    return {
        level: matrix[indices].sum(axis=0).astype(int).tolist()
        for level, matrix in positive_matrices.items()
    }


def create_screening_subset(
    table,
    size: int,
    seed: int,
    min_positive_per_concept: int = 200,
):
    """Select rare-label coverage first, then uniformly fill the remaining slots."""
    if not 0 < size <= table.num_rows:
        raise ValueError(f"subset size must be in [1,{table.num_rows}], got {size}")
    rng = np.random.default_rng(seed)
    positives = {
        level: _image_positive_matrix(table, target_field, mask_field)
        for level, target_field, mask_field in LEVELS
    }
    selected = set()
    candidates = []
    for level, matrix in positives.items():
        for concept_index in range(matrix.shape[1]):
            indices = np.flatnonzero(matrix[:, concept_index])
            candidates.append((len(indices), level, concept_index, indices))

    # Rarest labels select first so common labels cannot consume their capacity.
    for _, _, _, indices in sorted(candidates, key=lambda item: item[0]):
        available = np.asarray([index for index in indices if index not in selected])
        if not len(available):
            continue
        take = min(min_positive_per_concept, len(available), size - len(selected))
        if take:
            selected.update(rng.choice(available, size=take, replace=False).tolist())
        if len(selected) == size:
            break

    if len(selected) < size:
        remaining = np.asarray([index for index in range(table.num_rows) if index not in selected])
        selected.update(rng.choice(remaining, size=size - len(selected), replace=False).tolist())
    indices = np.asarray(sorted(selected), dtype=np.int64)
    return indices, {
        "strategy": "rare_positive_quota_then_uniform_fill_v1",
        "seed": int(seed),
        "size": int(size),
        "min_positive_per_concept": int(min_positive_per_concept),
        "full_positive_images": _coverage(positives, np.arange(table.num_rows)),
        "subset_positive_images": _coverage(positives, indices),
    }


def load_or_create_screening_subset(
    table,
    path: str | Path,
    size: int,
    seed: int,
    min_positive_per_concept: int = 200,
):
    path = Path(path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["size"] != size or payload["seed"] != seed:
            raise ValueError("Existing screening subset size/seed does not match config")
        return np.asarray(payload["indices"], dtype=np.int64), payload
    indices, report = create_screening_subset(
        table, size, seed, min_positive_per_concept
    )
    payload = {**report, "indices": indices.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return indices, payload
