"""Build phrase-aligned C1 texture observations from flattened annotations.

Texture cues are aligned to the same phrase index as anatomical findings. A
cue inherits polarity only from explicitly compatible findings in that phrase;
ambiguous or unmatched cues remain unknown (-1). Region-level conflicts are
rolled up in phrase order using the last explicit assertion.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import pandas as pd

DEFAULT_MAPPING = {
    "opacity": ["lung opacity", "airspace opacity", "pulmonary edema/hazy opacity"],
    "airspace": ["airspace opacity"],
    "interstitial": ["increased reticular markings/ild pattern"],
    "lucency": ["hyperaeration"],
}

def parse_parts(value):
    if not isinstance(value, str): return None
    parts = value.split("|", 2)
    if len(parts) != 3 or not parts[2].strip(): return None
    return parts[0].strip(), parts[1].strip().lower(), parts[2].strip()

def parse_groups(value):
    try: value = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError: return []
    if not isinstance(value, list): return []
    return [g if isinstance(g, list) else [g] if isinstance(g, str) else [] for g in value]

def slug(value):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_"))

def build(args):
    frame = pd.read_parquet(args.atomic_annotations, columns=[
        "dicom_id", "study_id", "subject_id", "region", "region_raw", "phrase_id",
        "category", "raw_label", "relation", "source_file", "texture_cues"])
    mapping = DEFAULT_MAPPING
    if args.texture_finding_mapping:
        table = pd.read_csv(args.texture_finding_mapping)
        mapping = {}
        for row in table.itertuples(index=False):
            mapping.setdefault(str(row.texture), []).append(str(row.finding))
    finding_lookup = {}
    for row in frame[frame.category.eq("anatomicalfinding")].itertuples(index=False):
        key = (row.dicom_id, row.region, int(row.phrase_id))
        finding_lookup.setdefault(key, []).append((str(row.raw_label), str(row.relation).lower()))
    phrase_rows = []
    for row in frame.drop_duplicates(["dicom_id", "region", "phrase_id", "texture_cues"]).itertuples(index=False):
        groups = parse_groups(row.texture_cues)
        cues = groups[int(row.phrase_id)] if int(row.phrase_id) < len(groups) else []
        for raw in cues:
            parsed = parse_parts(raw)
            if not parsed or parsed[0] != "texture": continue
            _, cue_relation, texture = parsed
            canonical = slug(texture)
            compatible = set(mapping.get(texture, []))
            candidates = [(label, pol) for label, pol in finding_lookup.get((row.dicom_id, row.region, int(row.phrase_id)), []) if label in compatible]
            polarities = {pol for _, pol in candidates if pol in {"yes", "no"}}
            state = next(iter(polarities)) if len(polarities) == 1 else "unknown"
            phrase_rows.append({"dicom_id": row.dicom_id, "study_id": row.study_id, "subject_id": row.subject_id,
                "region": row.region, "region_raw": row.region_raw, "phrase_id": int(row.phrase_id),
                "texture_raw": texture, "canonical": canonical, "cue_relation": cue_relation,
                "state": state, "matched_findings": json.dumps(candidates, ensure_ascii=False),
                "source_file": row.source_file})
    phrase = pd.DataFrame(phrase_rows)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    phrase.to_parquet(out / "c1_texture_phrase_annotations.parquet", index=False)
    if phrase.empty:
        pd.DataFrame(columns=["dicom_id","region","canonical","state"]).to_parquet(out / "c1_texture_region_labels.parquet", index=False)
        return
    # Keep the last explicit yes/no assertion in report order; unknown never
    # invents a state and is retained when no explicit assertion exists.
    phrase = phrase.sort_values(["dicom_id", "region", "phrase_id"])
    def rollup(group):
        explicit = group[group.state.isin(["yes", "no"])]
        row = (explicit.iloc[-1] if not explicit.empty else group.iloc[-1]).copy()
        row["state"] = row.state if row.state in {"yes", "no"} else "unknown"
        row["dicom_id"], row["region"], row["canonical"] = group.name
        row["phrase_count"] = len(group)
        return row
    region = phrase.groupby(["dicom_id", "region", "canonical"], sort=False, group_keys=False).apply(rollup).reset_index(drop=True)
    region.to_parquet(out / "c1_texture_region_labels.parquet", index=False)
    stats = region.groupby("canonical").agg(
        positive_count=("state", lambda s: int((s == "yes").sum())),
        negative_count=("state", lambda s: int((s == "no").sum())),
        unknown_count=("state", lambda s: int((s == "unknown").sum())),
        image_count=("dicom_id", "nunique"), region_count=("region", "nunique"),
    ).reset_index()
    stats.to_csv(out / "c1_texture_statistics.csv", index=False)
    with (out / "c1_build_summary.json").open("w") as f:
        json.dump({"phrase_rows": len(phrase), "region_labels": len(region), "concepts": int(region.canonical.nunique()),
                   "images": int(region.dicom_id.nunique()), "mapping": mapping}, f, indent=2)
    print(json.dumps({"phrase_rows": len(phrase), "region_labels": len(region), "concepts": int(region.canonical.nunique()), "images": int(region.dicom_id.nunique())}, indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--atomic-annotations", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--texture-finding-mapping")
    build(p.parse_args())
if __name__ == "__main__": main()
