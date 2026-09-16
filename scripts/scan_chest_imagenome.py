"""Flatten Chest ImaGenome scene graphs for annotation auditing.

This script deliberately stops before ontology mapping and tensor creation. It
keeps one row per raw attribute mention so that C1/C2/C3 decisions remain
auditable.

Example (after extracting the nested ``scene_graph.zip``)::

    python scripts/scan_chest_imagenome.py \
      --scene-graph-dir /media/user/Data/ChestImaGenome/scene_graph \
      --mimic-split /media/user/Data/mimic/MIMIC/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-split.csv.gz \
      --output-dir /media/user/Data/ChestImaGenome/processed/audit
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ATTRIBUTE_RE = re.compile(r"^([^|]+)\|([^|]+)\|(.*)$")


def _parse_attribute(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, str):
        return None
    match = ATTRIBUTE_RE.match(value.strip())
    if not match:
        return None
    category, relation, label = ((part or "").strip() for part in match.groups())
    if not category or not label:
        return None
    relation = relation or "unknown"
    return category, relation, label


def _iter_groups(value: Any) -> Iterable[list[str]]:
    """Yield attribute groups while tolerating minor release-format variants."""
    if not isinstance(value, list):
        return
    for group in value:
        if isinstance(group, list):
            yield [item for item in group if isinstance(item, str)]
        elif isinstance(group, str):
            yield [group]


def _canonical_region(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    replacements = {
        "right_lung": "right_lung",
        "left_lung": "left_lung",
        "right_upper_lung_zone": "right_upper_lung_zone",
        "right_mid_lung_zone": "right_mid_lung_zone",
        "right_lower_lung_zone": "right_lower_lung_zone",
        "left_upper_lung_zone": "left_upper_lung_zone",
        "left_mid_lung_zone": "left_mid_lung_zone",
        "left_lower_lung_zone": "left_lower_lung_zone",
    }
    return replacements.get(value, value)


def _read_split(path: Path) -> pd.DataFrame:
    split = pd.read_csv(path, compression="infer")
    required = {"dicom_id", "study_id", "subject_id", "split"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"MIMIC split is missing columns: {sorted(missing)}")
    split = split.copy()
    for col in ("dicom_id", "study_id", "subject_id", "split"):
        split[col] = split[col].astype(str)
    return split[["dicom_id", "study_id", "subject_id", "split"]].drop_duplicates("dicom_id")


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        pd.DataFrame().to_csv(path.with_suffix(".csv"), index=False)
        return
    frame = pd.DataFrame(rows)
    try:
        frame.to_parquet(path, index=False)
    except (ImportError, ModuleNotFoundError):
        fallback = path.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        print(f"pyarrow/fastparquet unavailable; wrote {fallback}")


def scan(args: argparse.Namespace) -> None:
    split = _read_split(Path(args.mimic_split))
    split_map = split.set_index("dicom_id").to_dict("index")
    json_files = sorted(Path(args.scene_graph_dir).glob("*.json"))
    if args.max_files:
        json_files = json_files[: args.max_files]
    if not json_files:
        raise FileNotFoundError(f"No scene graph JSON files found in {args.scene_graph_dir}")

    atomics: list[dict[str, Any]] = []
    bboxes: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    examples: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    stats: Counter[tuple[str, str, str]] = Counter()
    region_stats: Counter[tuple[str, str]] = Counter()
    seen_dicom: set[str] = set()
    malformed_files: list[dict[str, str]] = []

    for index, path in enumerate(json_files, 1):
        try:
            with path.open("r", encoding="utf-8") as handle:
                graph = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            # Public releases can contain empty/truncated members. Keep the
            # batch auditable while allowing valid graphs to be processed.
            malformed_files.append({"path": str(path), "error": type(exc).__name__})
            continue
        if not isinstance(graph, dict):
            malformed_files.append({"path": str(path), "error": "non_object_json"})
            continue
        dicom_id = str(graph.get("image_id") or path.name.split("_SceneGraph", 1)[0])
        if dicom_id not in split_map:
            continue
        meta = split_map[dicom_id]
        seen_dicom.add(dicom_id)
        study_id = str(graph.get("study_id", meta["study_id"]))
        subject_id = str(graph.get("patient_id", meta["subject_id"]))

        object_by_name: dict[str, dict[str, Any]] = {}
        for obj in graph.get("objects", []):
            if not isinstance(obj, dict) or not obj.get("bbox_name"):
                continue
            raw_region = str(obj["bbox_name"])
            region = _canonical_region(raw_region)
            object_by_name[raw_region] = obj
            bboxes.append({
                "dicom_id": dicom_id, "study_id": study_id, "subject_id": subject_id,
                "region_raw": raw_region, "region": region,
                "x1": obj.get("original_x1", obj.get("x1")),
                "y1": obj.get("original_y1", obj.get("y1")),
                "x2": obj.get("original_x2", obj.get("x2")),
                "y2": obj.get("original_y2", obj.get("y2")),
                "image_width": obj.get("original_width"),
                "image_height": obj.get("original_height"),
                "bbox_name": raw_region,
            })

        for attr in graph.get("attributes", []):
            if not isinstance(attr, dict):
                continue
            raw_region = str(attr.get("bbox_name") or attr.get("name") or "global")
            region = _canonical_region(raw_region)
            groups = attr.get("attributes", [])
            for phrase_id, group in enumerate(_iter_groups(groups)):
                for raw in group:
                    parsed = _parse_attribute(raw)
                    if parsed is None:
                        continue
                    category, relation, label = parsed
                    row = {
                        "dicom_id": dicom_id, "study_id": study_id, "subject_id": subject_id,
                        "region_raw": raw_region, "region": region,
                        "category": category, "raw_label": label, "relation": relation,
                        "phrase_id": phrase_id, "source_file": str(path),
                        "texture_cues": json.dumps(attr.get("texture_cues", []), ensure_ascii=False),
                        "severity_cues": json.dumps(attr.get("severity_cues", []), ensure_ascii=False),
                        "temporal_cues": json.dumps(attr.get("temporal_cues", []), ensure_ascii=False),
                    }
                    atomics.append(row)
                    stats[(category, label, relation)] += 1
                    region_stats[(region, category)] += 1
                    key = (category, label, relation)
                    if len(examples[key]) < args.examples_per_label:
                        examples[key].append(row)

        for rel in graph.get("relationships", []):
            if isinstance(rel, dict):
                relation_rows.append({
                    "dicom_id": dicom_id, "study_id": study_id, "subject_id": subject_id,
                    "relationship_id": rel.get("relationship_id"),
                    "relationship_names": json.dumps(rel.get("relationship_names", []), ensure_ascii=False),
                    "phrase": rel.get("phrase", ""),
                    "source_file": str(path),
                })
        if index % 10000 == 0:
            print(f"scanned {index}/{len(json_files)} scene graphs")

    stat_rows = []
    for (category, label, relation), count in sorted(stats.items()):
        stat_rows.append({
            "category": category, "raw_label": label, "relation": relation,
            "count": count,
            "study_count": len({r["study_id"] for r in atomics if r["category"] == category and r["raw_label"] == label and r["relation"] == relation}),
            "region_count": len({r["region"] for r in atomics if r["category"] == category and r["raw_label"] == label and r["relation"] == relation}),
        })
    region_rows = [{"region": r, "category": c, "count": n} for (r, c), n in sorted(region_stats.items())]
    example_rows = [row for key in sorted(examples) for row in examples[key]]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_table(atomics, out / "atomic_annotations.parquet")
    _write_table(bboxes, out / "anatomy_bboxes.parquet")
    _write_table(relation_rows, out / "comparison_relations.parquet")
    _write_table(stat_rows, out / "label_statistics.parquet")
    _write_table(region_rows, out / "region_statistics.parquet")
    _write_table(example_rows, out / "raw_label_examples.parquet")
    with (out / "scan_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "scene_graph_files_seen": len(json_files),
            "matched_mimic_images": len(seen_dicom),
            "atomic_annotation_rows": len(atomics),
            "bbox_rows": len(bboxes),
            "relationship_rows": len(relation_rows),
            "malformed_scene_graph_files": len(malformed_files),
            "malformed_scene_graph_examples": malformed_files[:20],
            "splits": split[split.dicom_id.isin(seen_dicom)]["split"].value_counts().to_dict(),
        }, handle, indent=2)
    print(json.dumps({"matched_mimic_images": len(seen_dicom), "atomic_rows": len(atomics), "output_dir": str(out)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-graph-dir", required=True, help="Directory containing *_SceneGraph.json files")
    parser.add_argument("--mimic-split", required=True, help="Official MIMIC split CSV(.gz)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-files", type=int, default=0, help="Only scan the first N JSON files")
    parser.add_argument("--examples-per-label", type=int, default=10)
    args = parser.parse_args()
    scan(args)


if __name__ == "__main__":
    main()
