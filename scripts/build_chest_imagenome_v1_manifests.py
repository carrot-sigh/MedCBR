"""Build frozen Chest ImaGenome V1 train/validate/test manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


C1_CONCEPTS = ["opacity", "calcified", "alveolar", "lucency", "interstitial"]
C1_QUALITY = ["high", "high", "medium", "medium", "low"]
C1_QUALITY_WEIGHT = np.asarray([1.0, 1.0, 0.75, 0.75, 0.5], dtype=np.float32)
SEVERITY_CONCEPTS = ["mild", "moderate", "severe", "hedge"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--mimic-split", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def nested_array(values: np.ndarray) -> pa.Array:
    """Convert [N, R, K] numpy data to a nested fixed-size Arrow list."""
    inner = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[2])
    return pa.FixedSizeListArray.from_arrays(inner, values.shape[1])


def vector_array(values: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])


def state_arrays(count: int, regions: int, concepts: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((count, regions, concepts), -1, dtype=np.int8),
        np.zeros((count, regions, concepts), dtype=np.bool_),
    )


def parse_cue_group(value: str, phrase_id: int, category: str) -> list[tuple[str, str]]:
    try:
        groups = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(groups, list) or phrase_id >= len(groups):
        return []
    group = groups[phrase_id]
    if isinstance(group, str):
        group = [group]
    if not isinstance(group, list):
        return []
    parsed = []
    for raw in group:
        if not isinstance(raw, str):
            continue
        parts = raw.split("|", 2)
        if len(parts) == 3 and parts[0].strip() == category:
            parsed.append((parts[2].strip(), parts[1].strip().lower()))
    return parsed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_path = args.audit_dir / "atomic_annotations.parquet"

    split = pd.read_csv(args.mimic_split, compression="infer", dtype=str)
    split = split[["dicom_id", "study_id", "subject_id", "split"]].drop_duplicates("dicom_id")
    bbox = pd.read_parquet(args.audit_dir / "anatomy_bboxes.parquet")
    image_ids = bbox["dicom_id"].drop_duplicates()
    base = split[split.dicom_id.isin(image_ids)].copy().sort_values("dicom_id").reset_index(drop=True)
    base["image_path"] = base.apply(
        lambda row: str(
            args.image_root
            / f"p{row.subject_id[:2]}"
            / f"p{row.subject_id}"
            / f"s{row.study_id}"
            / f"{row.dicom_id}.jpg"
        ),
        axis=1,
    )
    base["image_exists"] = base.image_path.map(lambda value: Path(value).is_file())
    if not base.image_exists.all():
        missing = base.loc[~base.image_exists, "image_path"].head(10).tolist()
        raise FileNotFoundError(f"Missing {int((~base.image_exists).sum())} JPGs; examples: {missing}")

    regions = pd.read_csv(args.audit_dir / "hierarchy_audit" / "region_vocabulary.csv")["region"].tolist()
    if len(regions) != 32:
        raise ValueError(f"Frozen V1 requires 32 target regions, found {len(regions)}")
    c2_concepts = sorted(
        pd.read_parquet(atomic_path, filters=[("category", "==", "anatomicalfinding")], columns=["raw_label"])
        ["raw_label"].drop_duplicates().tolist()
    )
    c3_concepts = sorted(
        pd.read_parquet(atomic_path, filters=[("category", "==", "disease")], columns=["raw_label"])
        ["raw_label"].drop_duplicates().tolist()
    )
    if (len(c2_concepts), len(c3_concepts)) != (43, 10):
        raise ValueError(f"Unexpected V1 vocabulary sizes: C2={len(c2_concepts)}, C3={len(c3_concepts)}")

    n, r = len(base), len(regions)
    image_index = pd.Series(np.arange(n), index=base.dicom_id)
    region_index = {name: index for index, name in enumerate(regions)}
    c1_index = {name: index for index, name in enumerate(C1_CONCEPTS)}
    c2_index = {name: index for index, name in enumerate(c2_concepts)}
    c3_index = {name: index for index, name in enumerate(c3_concepts)}
    severity_index = {name: index for index, name in enumerate(SEVERITY_CONCEPTS)}

    # Bounding boxes use absolute scene-graph coordinates. Seven annotation-only
    # target regions have no bbox and therefore retain bbox_mask=False.
    region_bboxes = np.zeros((n, r, 4), dtype=np.float32)
    region_bbox_mask = np.zeros((n, r), dtype=np.bool_)
    bbox = bbox[bbox.region.isin(region_index)].drop_duplicates(["dicom_id", "region"], keep="last")
    bbox = bbox[bbox.dicom_id.isin(image_index.index)]
    bi = bbox.dicom_id.map(image_index).to_numpy()
    br = bbox.region.map(region_index).to_numpy()
    region_bboxes[bi, br] = bbox[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    region_bbox_mask[bi, br] = True

    c1_target, c1_mask = state_arrays(n, r, len(C1_CONCEPTS))
    c1 = pd.read_parquet(args.audit_dir / "c1" / "c1_texture_region_labels.parquet")
    c1 = c1[c1.dicom_id.isin(image_index.index) & c1.region.isin(region_index) & c1.canonical.isin(c1_index)]
    explicit = c1.state.isin(["yes", "no"])
    c1 = c1[explicit]
    ci, cr, cc = c1.dicom_id.map(image_index).to_numpy(), c1.region.map(region_index).to_numpy(), c1.canonical.map(c1_index).to_numpy()
    c1_target[ci, cr, cc] = c1.state.eq("yes").to_numpy(dtype=np.int8)
    c1_mask[ci, cr, cc] = True

    c2_target, c2_mask = state_arrays(n, r, len(c2_concepts))
    c2 = pd.read_parquet(
        atomic_path,
        filters=[("category", "==", "anatomicalfinding")],
        columns=["dicom_id", "region", "raw_label", "relation", "phrase_id"],
    )
    c2 = c2[c2.dicom_id.isin(image_index.index) & c2.region.isin(region_index) & c2.relation.isin(["yes", "no"])]
    c2 = c2.sort_values("phrase_id").drop_duplicates(["dicom_id", "region", "raw_label"], keep="last")
    i2, r2, k2 = c2.dicom_id.map(image_index).to_numpy(), c2.region.map(region_index).to_numpy(), c2.raw_label.map(c2_index).to_numpy()
    c2_target[i2, r2, k2] = c2.relation.eq("yes").to_numpy(dtype=np.int8)
    c2_mask[i2, r2, k2] = True
    del c2

    severity_target, severity_mask = state_arrays(n, r, len(SEVERITY_CONCEPTS))
    cues = pd.read_parquet(
        atomic_path,
        columns=["dicom_id", "region", "phrase_id", "severity_cues"],
    )
    cues = cues[cues.severity_cues.str.contains("severity|", regex=False, na=False)]
    cues = cues.drop_duplicates(["dicom_id", "region", "phrase_id", "severity_cues"])
    severity_rows: list[tuple[str, str, int, str, str]] = []
    for row in cues.itertuples(index=False):
        for concept, relation in parse_cue_group(row.severity_cues, int(row.phrase_id), "severity"):
            if concept in severity_index and relation in {"yes", "no"}:
                severity_rows.append((row.dicom_id, row.region, int(row.phrase_id), concept, relation))
    severity = pd.DataFrame(severity_rows, columns=["dicom_id", "region", "phrase_id", "concept", "relation"])
    severity = severity[severity.dicom_id.isin(image_index.index) & severity.region.isin(region_index)]
    severity = severity.sort_values("phrase_id").drop_duplicates(["dicom_id", "region", "concept"], keep="last")
    si, sr, sk = severity.dicom_id.map(image_index).to_numpy(), severity.region.map(region_index).to_numpy(), severity.concept.map(severity_index).to_numpy()
    severity_target[si, sr, sk] = severity.relation.eq("yes").to_numpy(dtype=np.int8)
    severity_mask[si, sr, sk] = True

    c3_target = np.full((n, len(c3_concepts)), -1, dtype=np.int8)
    c3_mask = np.zeros((n, len(c3_concepts)), dtype=np.bool_)
    c3 = pd.read_parquet(
        atomic_path,
        filters=[("category", "==", "disease")],
        columns=["dicom_id", "raw_label", "relation"],
    ).drop_duplicates()
    polarity = c3.groupby(["dicom_id", "raw_label"]).relation.agg(
        has_yes=lambda values: bool((values == "yes").any()),
        has_no=lambda values: bool((values == "no").any()),
    ).reset_index()
    polarity = polarity[polarity.has_yes ^ polarity.has_no]
    polarity = polarity[polarity.dicom_id.isin(image_index.index)]
    i3, k3 = polarity.dicom_id.map(image_index).to_numpy(), polarity.raw_label.map(c3_index).to_numpy()
    c3_target[i3, k3] = polarity.has_yes.to_numpy(dtype=np.int8)
    c3_mask[i3, k3] = True

    c1_active_regions = np.asarray([region in set(c1.region) for region in regions], dtype=np.bool_)
    arrays = {
        "region_bboxes": nested_array(region_bboxes),
        "region_bbox_mask": vector_array(region_bbox_mask),
        "c1_target": nested_array(c1_target), "c1_mask": nested_array(c1_mask),
        "c2_target": nested_array(c2_target), "c2_mask": nested_array(c2_mask),
        "severity_target": nested_array(severity_target), "severity_mask": nested_array(severity_mask),
        "c3_target": vector_array(c3_target), "c3_mask": vector_array(c3_mask),
        "c1_quality_weight": vector_array(np.broadcast_to(C1_QUALITY_WEIGHT, (n, len(C1_CONCEPTS))).copy()),
        "region_active_c1": vector_array(np.broadcast_to(c1_active_regions, (n, r)).copy()),
    }
    table = pa.table({
        "dicom_id": pa.array(base.dicom_id), "study_id": pa.array(base.study_id),
        "subject_id": pa.array(base.subject_id), "image_path": pa.array(base.image_path),
        "image_exists": pa.array(base.image_exists), "split": pa.array(base.split), **arrays,
    })
    split_names = {"train": "train.parquet", "validate": "valid.parquet", "test": "test.parquet"}
    counts = {}
    for split_name, filename in split_names.items():
        indices = np.flatnonzero(base.split.eq(split_name).to_numpy())
        pq.write_table(table.take(pa.array(indices)), args.output_dir / filename, compression="zstd")
        counts[split_name] = len(indices)

    bbox_regions = sorted(pd.read_parquet(args.audit_dir / "anatomy_bboxes.parquet", columns=["region"]).region.unique())
    metadata = {
        "version": "chest_imagenome_hierarchy_v1",
        "shapes": {"regions": 32, "c1": 5, "c2": 43, "severity": 4, "c3": 10},
        "regions": regions, "c1_concepts": C1_CONCEPTS, "c1_quality": C1_QUALITY,
        "c1_quality_weight": C1_QUALITY_WEIGHT.tolist(), "c2_concepts": c2_concepts,
        "severity_concepts": SEVERITY_CONCEPTS, "c3_concepts": c3_concepts,
        "bbox_regions_in_source": bbox_regions,
        "target_regions_without_bbox": sorted(set(regions) - set(bbox_regions)),
        "bbox_regions_excluded_from_targets": sorted(set(bbox_regions) - set(regions)),
        "label_encoding": {"present": 1, "explicitly_absent": 0, "unknown_target": -1, "unknown_mask": False},
        "rollup": {
            "c1_c2_severity": "last explicit assertion by phrase_id within image + region + concept",
            "c3": "unanimous explicit polarity across image; conflicting yes/no is unknown",
        },
        "circularity_note": (
            "C1 polarity is weakly supervised using compatible C2 finding assertions. "
            "C1-C2 association statistics audit extraction consistency and are not independent evidence of causal hierarchy."
        ),
        "image_root": str(args.image_root), "all_images_verified": True, "split_counts": counts,
        "severity_phrase_rows": len(severity_rows), "severity_region_labels": len(severity),
    }
    with (args.output_dir / "ontology_v1.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (args.output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "rows": n, "split_counts": counts,
            "explicit_labels": {
                "c1": int(c1_mask.sum()), "c2": int(c2_mask.sum()),
                "severity": int(severity_mask.sum()), "c3": int(c3_mask.sum()),
            },
            "bbox_slots": int(region_bbox_mask.sum()), "missing_images": int((~base.image_exists).sum()),
        }, handle, indent=2)
    print(json.dumps(metadata["shapes"] | {"rows": n, "split_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
