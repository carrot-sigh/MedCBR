"""Audit frozen V1 manifests before model training."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def nested_numpy(array, rows: int, concepts: int) -> np.ndarray:
    return array.values.values.to_numpy(zero_copy_only=False).reshape(-1, rows, concepts)


def vector_numpy(array, concepts: int) -> np.ndarray:
    return array.values.to_numpy(zero_copy_only=False).reshape(-1, concepts)


def parse_cues(group, category: str) -> list[tuple[str, str]]:
    if isinstance(group, str):
        group = [group]
    if not isinstance(group, list):
        return []
    result = []
    for raw in group:
        if not isinstance(raw, str):
            continue
        parts = raw.split("|", 2)
        if len(parts) == 3 and parts[0].strip() == category:
            result.append((parts[2].strip(), parts[1].strip().lower()))
    return result


def normalize_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def label_and_density_audit(manifest_dir: Path, ontology: dict, output_dir: Path) -> dict:
    regions = ontology["regions"]
    specs = {
        "c1": (ontology["c1_concepts"], 32),
        "c2": (ontology["c2_concepts"], 32),
        "c3": (ontology["c3_concepts"], None),
    }
    counts = {}
    patients = {
        layer: {state: [set() for _ in concepts] for state in ("positive", "negative", "unknown")}
        for layer, (concepts, _) in specs.items()
    }
    density = {
        layer: {
            "positive": np.zeros((rows, len(concepts)), dtype=np.int64),
            "negative": np.zeros((rows, len(concepts)), dtype=np.int64),
            "known": np.zeros((rows, len(concepts)), dtype=np.int64),
        }
        for layer, (concepts, rows) in specs.items() if rows is not None
    }
    total_images = 0
    for split in ("train", "valid", "test"):
        parquet = pq.ParquetFile(manifest_dir / f"{split}.parquet")
        columns = ["subject_id"] + [item for layer in specs for item in (f"{layer}_target", f"{layer}_mask")]
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            total_images += batch.num_rows
            subjects = np.asarray(batch.column("subject_id").to_pylist(), dtype=object)
            for layer, (concepts, rows) in specs.items():
                target_col = batch.column(f"{layer}_target")
                mask_col = batch.column(f"{layer}_mask")
                if rows is None:
                    target = vector_numpy(target_col, len(concepts))
                    mask = vector_numpy(mask_col, len(concepts)).astype(bool)
                    axes = None
                else:
                    target = nested_numpy(target_col, rows, len(concepts))
                    mask = nested_numpy(mask_col, rows, len(concepts)).astype(bool)
                    axes = 1
                    density[layer]["positive"] += ((target == 1) & mask).sum(axis=0)
                    density[layer]["negative"] += ((target == 0) & mask).sum(axis=0)
                    density[layer]["known"] += mask.sum(axis=0)
                for concept_index in range(len(concepts)):
                    concept_target = target[..., concept_index]
                    concept_mask = mask[..., concept_index]
                    positive_image = ((concept_target == 1) & concept_mask)
                    negative_image = ((concept_target == 0) & concept_mask)
                    unknown_image = ~concept_mask
                    if axes is not None:
                        positive_image = positive_image.any(axis=axes)
                        negative_image = negative_image.any(axis=axes)
                        unknown_image = unknown_image.any(axis=axes)
                    patients[layer]["positive"][concept_index].update(subjects[positive_image])
                    patients[layer]["negative"][concept_index].update(subjects[negative_image])
                    patients[layer]["unknown"][concept_index].update(subjects[unknown_image])
                layer_counts = counts.setdefault(layer, {
                    "positive": np.zeros(len(concepts), dtype=np.int64),
                    "negative": np.zeros(len(concepts), dtype=np.int64),
                    "unknown": np.zeros(len(concepts), dtype=np.int64),
                })
                reduce_axes = tuple(range(target.ndim - 1))
                layer_counts["positive"] += ((target == 1) & mask).sum(axis=reduce_axes)
                layer_counts["negative"] += ((target == 0) & mask).sum(axis=reduce_axes)
                layer_counts["unknown"] += (~mask).sum(axis=reduce_axes)

    rows = []
    for layer, (concepts, _) in specs.items():
        for index, concept in enumerate(concepts):
            row = {"layer": layer.upper(), "concept": concept}
            for state in ("positive", "negative", "unknown"):
                row[f"{state}_count"] = int(counts[layer][state][index])
                row[f"{state}_patient_count"] = len(patients[layer][state][index])
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "concept_label_counts.csv", index=False)

    for layer in ("c1", "c2"):
        concepts = specs[layer][0]
        records = []
        for region_index, region in enumerate(regions):
            for concept_index, concept in enumerate(concepts):
                known = int(density[layer]["known"][region_index, concept_index])
                records.append({
                    "region": region, "concept": concept,
                    "positive_count": int(density[layer]["positive"][region_index, concept_index]),
                    "negative_count": int(density[layer]["negative"][region_index, concept_index]),
                    "unknown_count": total_images - known,
                    "mask_density": known / total_images,
                })
        pd.DataFrame(records).to_csv(output_dir / f"{layer}_region_mask_density.csv", index=False)
    return {"images": total_images}


def conditional_matrix(manifest_dir: Path, ontology: dict, output_dir: Path) -> dict:
    c1_names, c2_names = ontology["c1_concepts"], ontology["c2_concepts"]
    numerator = np.zeros((len(c1_names), len(c2_names)), dtype=np.int64)
    denominator = np.zeros(len(c1_names), dtype=np.int64)
    for split in ("train", "valid", "test"):
        parquet = pq.ParquetFile(manifest_dir / f"{split}.parquet")
        for batch in parquet.iter_batches(batch_size=4096, columns=["c1_target", "c1_mask", "c2_target", "c2_mask"]):
            c1 = nested_numpy(batch.column("c1_target"), 32, len(c1_names))
            c1m = nested_numpy(batch.column("c1_mask"), 32, len(c1_names)).astype(bool)
            c2 = nested_numpy(batch.column("c2_target"), 32, len(c2_names))
            c2m = nested_numpy(batch.column("c2_mask"), 32, len(c2_names)).astype(bool)
            c1p = (c1 == 1) & c1m
            c2p = (c2 == 1) & c2m
            denominator += c1p.sum(axis=(0, 1))
            numerator += np.einsum("nri,nrj->ij", c1p.astype(np.int64), c2p.astype(np.int64))
    probabilities = np.divide(numerator, denominator[:, None], out=np.zeros_like(numerator, dtype=float), where=denominator[:, None] > 0)
    pd.DataFrame(probabilities, index=c1_names, columns=c2_names).rename_axis("c1_concept").to_csv(output_dir / "p_c2_positive_given_c1_positive.csv")
    pd.DataFrame(numerator, index=c1_names, columns=c2_names).rename_axis("c1_concept").to_csv(output_dir / "c1_c2_positive_support.csv")
    long = []
    for i, c1 in enumerate(c1_names):
        for j, c2 in enumerate(c2_names):
            long.append({"c1_concept": c1, "c2_concept": c2, "support_region_instances": int(numerator[i, j]), "c1_positive_region_instances": int(denominator[i]), "p_c2_given_c1": probabilities[i, j]})
    pd.DataFrame(long).sort_values(["c1_concept", "p_c2_given_c1"], ascending=[True, False]).to_csv(output_dir / "c1_c2_conditional_long.csv", index=False)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(18, 4.5))
        image = ax.imshow(probabilities, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(range(len(c1_names)), c1_names)
        ax.set_xticks(range(len(c2_names)), c2_names, rotation=90, fontsize=7)
        ax.set_title("P(C2 positive | C1 positive), same image-region")
        fig.colorbar(image, ax=ax, label="conditional probability")
        fig.tight_layout()
        fig.savefig(output_dir / "p_c2_given_c1_heatmap.png", dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    return {"c1_positive_region_instances": {name: int(value) for name, value in zip(c1_names, denominator)}}


def severity_audit(manifest_dir: Path, output_dir: Path, seed: int) -> dict:
    # Source files are preserved on C1 phrase rows and point to the same scene graphs.
    audit_root = manifest_dir.parents[1] / "chest_imagenome_audit_full"
    atomic = pd.read_parquet(audit_root / "atomic_annotations.parquet", columns=["source_file", "severity_cues"])
    files = atomic.loc[atomic.severity_cues.str.contains("severity|", regex=False, na=False), "source_file"].drop_duplicates().tolist()
    del atomic
    groups: dict[tuple[str, str, str, str], dict] = {}
    for file_index, source in enumerate(files, 1):
        with Path(source).open("r", encoding="utf-8") as handle:
            graph = json.load(handle)
        dicom_id = str(graph.get("image_id") or Path(source).name.split("_SceneGraph", 1)[0])
        for attribute in graph.get("attributes", []):
            if not isinstance(attribute, dict):
                continue
            region = re.sub(r"[^a-z0-9]+", "_", str(attribute.get("bbox_name") or attribute.get("name") or "global").lower()).strip("_")
            cue_groups = attribute.get("severity_cues", [])
            phrases = attribute.get("phrases", [])
            phrase_ids = attribute.get("phrase_IDs", [])
            sections = attribute.get("sections", [])
            attribute_groups = attribute.get("attributes", [])
            for phrase_index, cue_group in enumerate(cue_groups if isinstance(cue_groups, list) else []):
                for concept, relation in parse_cues(cue_group, "severity"):
                    phrase = normalize_phrase(phrases[phrase_index] if phrase_index < len(phrases) else "")
                    phrase_id = str(phrase_ids[phrase_index]) if phrase_index < len(phrase_ids) else ""
                    phrase_key = phrase_id or phrase or f"local:{phrase_index}"
                    key = (dicom_id, phrase_key, concept, relation)
                    item = groups.setdefault(key, {"dicom_id": dicom_id, "source_file": source, "phrase_id": phrase_id,
                        "phrase": phrase, "section": str(sections[phrase_index]) if phrase_index < len(sections) else "",
                        "concept": concept, "relation": relation, "regions": set(), "region_findings": defaultdict(set)})
                    item["regions"].add(region)
                    group = attribute_groups[phrase_index] if phrase_index < len(attribute_groups) else []
                    for raw in group if isinstance(group, list) else []:
                        parts = raw.split("|", 2) if isinstance(raw, str) else []
                        if len(parts) == 3 and parts[0] == "anatomicalfinding":
                            item["region_findings"][region].add(f"{parts[2]}={parts[1]}")
        if file_index % 10000 == 0:
            print(f"severity source scan {file_index}/{len(files)}")
    records = []
    for item in groups.values():
        records.append({
            "dicom_id": item["dicom_id"], "source_file": item["source_file"], "phrase_id": item["phrase_id"],
            "section": item["section"], "concept": item["concept"], "relation": item["relation"], "phrase": item["phrase"],
            "region_count": len(item["regions"]), "regions": json.dumps(sorted(item["regions"])),
            "region_findings": json.dumps({key: sorted(value) for key, value in sorted(item["region_findings"].items())}, ensure_ascii=False),
        })
    expansion = pd.DataFrame(records)
    expansion.to_parquet(output_dir / "severity_expansion_provenance.parquet", index=False)
    sample = expansion.sample(n=min(100, len(expansion)), random_state=seed).sort_values(["concept", "region_count"], ascending=[True, False])
    sample.to_csv(output_dir / "severity_expansion_sample_100.csv", index=False)
    by_concept = expansion.groupby("concept").region_count.agg(["count", "mean", "median", lambda values: values.quantile(.9), lambda values: values.quantile(.99), "max"]).reset_index()
    by_concept.columns = ["concept", "raw_cue_count", "mean_regions", "p50_regions", "p90_regions", "p99_regions", "max_regions"]
    by_concept.to_csv(output_dir / "severity_expansion_by_concept.csv", index=False)
    overall = expansion.region_count.describe(percentiles=[.5, .9, .99]).to_dict()
    return {"source_files": len(files), "raw_cues": len(expansion), "region_labels": int(expansion.region_count.sum()), "regions_per_cue": {key: float(value) for key, value in overall.items()}}


def diagnosis_support(manifest_dir: Path, ontology: dict, output_dir: Path) -> dict:
    audit_root = manifest_dir.parents[1] / "chest_imagenome_audit_full"
    strength = pd.read_csv(audit_root / "hierarchy_audit" / "c2_c3_strength.csv")
    strength["rank_within_c3"] = strength.groupby("c3_concept").lift_c2_c3.rank(method="first", ascending=False).astype(int)
    strength.sort_values(["c3_concept", "rank_within_c3"]).to_csv(output_dir / "c3_c2_support_ranked.csv", index=False)
    diagnosis_tokens = {concept: set(re.findall(r"[a-z]+", concept.lower())) for concept in ontology["c3_concepts"]}
    leakage = []
    for layer, concepts in (("C1", ontology["c1_concepts"]), ("C2", ontology["c2_concepts"])):
        for concept in concepts:
            tokens = set(re.findall(r"[a-z]+", concept.lower()))
            for diagnosis, disease_tokens in diagnosis_tokens.items():
                overlap = sorted(tokens & disease_tokens)
                if overlap:
                    leakage.append({"layer": layer, "concept": concept, "c3_diagnosis": diagnosis, "shared_tokens": ",".join(overlap), "exact_match": concept.lower() == diagnosis.lower()})
    pd.DataFrame(leakage).to_csv(output_dir / "diagnosis_lexical_leakage_audit.csv", index=False)
    return {"c2_c3_pairs": len(strength), "exact_diagnosis_matches_in_c1_c2": sum(item["exact_match"] for item in leakage)}


def main() -> None:
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ontology = json.load((args.manifest_dir / "ontology_v1.json").open())
    summary = {
        "label_and_mask_audit": label_and_density_audit(args.manifest_dir, ontology, args.output_dir),
        "c1_c2_conditional": conditional_matrix(args.manifest_dir, ontology, args.output_dir),
        "severity_expansion": severity_audit(args.manifest_dir, args.output_dir, args.seed),
        "diagnosis_support": diagnosis_support(args.manifest_dir, ontology, args.output_dir),
        "interpretation": {
            "c1_c2": "Consistency/duplication audit only because C1 polarity uses compatible C2 assertions.",
            "c2_c3": "Natural same-study association; not causal evidence.",
        },
    }
    with (args.output_dir / "pretraining_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
