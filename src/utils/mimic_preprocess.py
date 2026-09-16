"""Build MedCBR MIMIC-CXR CSVs from official metadata and annotations.

Example:
python -m src.utils.mimic_preprocess \
  --images-root /data/mimic-cxr-jpg/2.0.0/files \
  --split /data/mimic-cxr-2.0.0-split.csv.gz \
  --metadata /data/mimic-cxr-2.0.0-metadata.csv.gz \
  --reports /data/mimic_reports.csv \
  --chexpert /data/mimic-cxr-2.0.0-chexpert.csv.gz \
  --states /data/radgraph_states.csv \
  --output-dir data/mimic
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from src.utils.clinical_states import parse_state_value


def _read(path):
    return pd.read_csv(path, compression="infer")


def _text_column(df, name):
    return df[name] if name in df else pd.Series([""] * len(df), index=df.index)


def _image_path(root: Path, subject_id, study_id, dicom_id):
    subject = str(int(subject_id))
    study = str(int(study_id))
    return str(root / f"p{subject[:2]}" / f"p{subject}" / f"s{study}" / f"{dicom_id}.jpg")


def build(args):
    split = _read(args.split)
    required = {"dicom_id", "study_id", "subject_id", "split"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"official split is missing columns: {sorted(missing)}")
    split = split[["dicom_id", "study_id", "subject_id", "split"]].copy()
    split["dicom_id"] = split["dicom_id"].astype(str)
    split["study_id"] = split["study_id"].astype(str)
    split["subject_id"] = split["subject_id"].astype(str)

    if args.metadata:
        meta = _read(args.metadata)
        cols = [c for c in ["dicom_id", "study_id", "subject_id", "ViewPosition"] if c in meta]
        split = split.merge(meta[cols].drop_duplicates("dicom_id"), on=["dicom_id"], how="left", suffixes=("", "_meta"))
    view = split.get("ViewPosition", pd.Series([""] * len(split))).fillna("").astype(str)

    reports = _read(args.reports)
    if "study_id" not in reports:
        raise ValueError("reports file requires study_id")
    report_cols = [c for c in ["study_id", "findings", "impression"] if c in reports]
    reports = reports[report_cols].copy()
    reports["study_id"] = reports["study_id"].astype(str)
    split = split.merge(reports.drop_duplicates("study_id"), on="study_id", how="left")

    if args.chexpert:
        labels = _read(args.chexpert)
        if "study_id" not in labels:
            raise ValueError("CheXpert file requires study_id")
        labels["study_id"] = labels["study_id"].astype(str)
        label_cols = [c for c in labels.columns if c not in {"study_id", "subject_id"}]
        split = split.merge(labels[["study_id"] + label_cols].drop_duplicates("study_id"), on="study_id", how="left")
        split["chexpert_labels"] = split[label_cols].fillna(-1).astype(float).apply(lambda r: r.to_dict(), axis=1).map(json.dumps)
        split = split.drop(columns=label_cols)
    else:
        split["chexpert_labels"] = "{}"

    if args.states:
        states = _read(args.states)
        if "study_id" not in states:
            raise ValueError("states file requires study_id")
        state_col = "clinical_states" if "clinical_states" in states else "radgraph_labels"
        if state_col not in states:
            raise ValueError("states file requires clinical_states or radgraph_labels")
        states["study_id"] = states["study_id"].astype(str)
        states = states[["study_id", state_col]].drop_duplicates("study_id").rename(columns={state_col: "radgraph_labels"})
        states["radgraph_labels"] = states["radgraph_labels"].map(lambda value: json.dumps(parse_state_value(value)))
        split = split.merge(states, on="study_id", how="left")
    else:
        split["radgraph_labels"] = "{}"

    split["image_path"] = [
        _image_path(Path(args.images_root), s, st, d)
        for s, st, d in zip(split.subject_id, split.study_id, split.dicom_id)
    ]
    split["view"] = view
    for col in ["findings", "impression", "radgraph_labels"]:
        if col not in split:
            split[col] = ""
        split[col] = split[col].fillna("").astype(str)
    split["text"] = split.apply(lambda r: json.dumps([r.findings, r.impression]), axis=1)
    split["text_augment"] = "[]"
    out_cols = ["dicom_id", "study_id", "subject_id", "image_path", "view", "findings", "impression", "text", "text_augment", "chexpert_labels", "radgraph_labels"]
    output = split[out_cols + ["split"]].copy()
    split_sets = {p: set(output.loc[output.split.str.lower() == p, "subject_id"]) for p in output.split.unique()}
    for left, left_ids in split_sets.items():
        for right, right_ids in split_sets.items():
            if left < right and left_ids & right_ids:
                raise ValueError(f"patient leakage between {left} and {right}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, name in [("train", "mimic_train.csv"), ("validate", "mimic_valid.csv"), ("valid", "mimic_valid.csv"), ("test", "mimic_test.csv")]:
        rows = output[output.split.str.lower() == part].drop(columns="split")
        if len(rows) and not (out_dir / name).exists():
            rows.to_csv(out_dir / name, index=False)
    print(f"wrote {len(output)} image rows to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--reports", required=True)
    parser.add_argument("--chexpert")
    parser.add_argument("--states")
    parser.add_argument("--output-dir", default="data/mimic")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
