"""PyTorch Dataset for the CSVs produced by ``mimic_preprocess``."""

import ast
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _parse(value):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return value


class MIMICMedCBRDataset(Dataset):
    """Image-level dataset with study-level reports, labels, and states."""

    def __init__(self, csv_path, image_size=224, train=False, normalize="imagenet"):
        self.df = pd.read_csv(csv_path)
        required = {"dicom_id", "study_id", "subject_id", "image_path", "view"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")
        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalize == "imagenet" else ([0.5] * 3, [0.5] * 3)
        ops = [transforms.Resize((image_size, image_size))]
        if train:
            ops += [transforms.RandomHorizontalFlip()]
        ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = self.transform(Image.open(row.image_path).convert("RGB"))
        labels = _parse(row.get("chexpert_labels", "{}"))
        states = _parse(row.get("radgraph_labels", "{}"))
        return {
            "img": image,
            "img_name": row.image_path,
            "dicom_id": str(row.dicom_id),
            "study_id": str(row.study_id),
            "subject_id": str(row.subject_id),
            "view": row.view,
            "findings": row.get("findings", ""),
            "impression": row.get("impression", ""),
            "report": row.get("text", ""),
            "label": labels,
            "concepts": states,
        }
