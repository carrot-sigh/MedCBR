"""Add patient-level and severity-noise checks to an existing V1 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def nested(array, regions, concepts):
    return array.values.values.to_numpy(zero_copy_only=False).reshape(-1, regions, concepts)


def vector(array, concepts):
    return array.values.to_numpy(zero_copy_only=False).reshape(-1, concepts)


def patient_counts(manifest_dir: Path, audit_dir: Path, ontology: dict) -> None:
    table = pd.read_csv(audit_dir / "concept_label_counts.csv")
    specs = {"C1": (ontology["c1_concepts"], 32), "C2": (ontology["c2_concepts"], 32), "C3": (ontology["c3_concepts"], None)}
    all_subjects = set()
    positive = {layer: [set() for _ in concepts] for layer, (concepts, _) in specs.items()}
    negative = {layer: [set() for _ in concepts] for layer, (concepts, _) in specs.items()}
    for split in ("train", "valid", "test"):
        columns = ["subject_id"] + [name for layer in specs for name in (f"{layer.lower()}_target", f"{layer.lower()}_mask")]
        for batch in pq.ParquetFile(manifest_dir / f"{split}.parquet").iter_batches(batch_size=4096, columns=columns):
            subjects = np.asarray(batch.column("subject_id").to_pylist(), dtype=object)
            all_subjects.update(subjects)
            for layer, (concepts, regions) in specs.items():
                prefix = layer.lower()
                if regions is None:
                    target = vector(batch.column(f"{prefix}_target"), len(concepts))
                    mask = vector(batch.column(f"{prefix}_mask"), len(concepts)).astype(bool)
                else:
                    target = nested(batch.column(f"{prefix}_target"), regions, len(concepts))
                    mask = nested(batch.column(f"{prefix}_mask"), regions, len(concepts)).astype(bool)
                for index in range(len(concepts)):
                    pos = ((target[..., index] == 1) & mask[..., index])
                    neg = ((target[..., index] == 0) & mask[..., index])
                    if regions is not None:
                        pos, neg = pos.any(axis=1), neg.any(axis=1)
                    positive[layer][index].update(subjects[pos])
                    negative[layer][index].update(subjects[neg])
    for layer, (concepts, _) in specs.items():
        for index, concept in enumerate(concepts):
            selector = table.layer.eq(layer) & table.concept.eq(concept)
            table.loc[selector, "positive_patient_count"] = len(positive[layer][index])
            table.loc[selector, "negative_patient_count"] = len(negative[layer][index])
            table.loc[selector, "unknown_patient_count"] = len(all_subjects - positive[layer][index] - negative[layer][index])
    table.to_csv(audit_dir / "concept_label_counts.csv", index=False)


def severity_flags(audit_dir: Path) -> dict:
    expansion = pd.read_parquet(audit_dir / "severity_expansion_provenance.parquet")
    phrase_keys = ["dicom_id", "phrase_id", "phrase"]
    phrase_concepts = expansion.groupby(phrase_keys, dropna=False).concept.nunique()
    region_rows = []
    for row in expansion.itertuples(index=False):
        findings = json.loads(row.region_findings)
        for region in json.loads(row.regions):
            values = findings.get(region, [])
            region_rows.append({
                "dicom_id": row.dicom_id, "phrase_id": row.phrase_id, "concept": row.concept,
                "region": region, "has_positive_finding": any(value.endswith("=yes") for value in values),
                "has_any_finding": bool(values),
            })
    region = pd.DataFrame(region_rows)
    region.to_parquet(audit_dir / "severity_region_quality_flags.parquet", index=False)
    return {
        "unique_phrases": len(phrase_concepts),
        "multi_severity_phrase_count": int((phrase_concepts > 1).sum()),
        "multi_severity_phrase_rate": float((phrase_concepts > 1).mean()),
        "region_labels_without_positive_finding": int((~region.has_positive_finding).sum()),
        "region_labels_without_positive_finding_rate": float((~region.has_positive_finding).mean()),
        "region_labels_without_any_finding": int((~region.has_any_finding).sum()),
        "region_labels_without_any_finding_rate": float((~region.has_any_finding).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    ontology = json.load((args.manifest_dir / "ontology_v1.json").open())
    patient_counts(args.manifest_dir, args.audit_dir, ontology)
    summary_path = args.audit_dir / "pretraining_audit_summary.json"
    summary = json.load(summary_path.open())
    summary["severity_expansion"]["quality_flags"] = severity_flags(args.audit_dir)
    summary["severity_expansion"]["recommendation"] = "Do not use severity in the primary V1 loss until modifier-to-finding alignment is resolved."
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["severity_expansion"], indent=2))


if __name__ == "__main__":
    main()
