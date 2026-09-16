"""Utilities for study-level clinical states used by the MIMIC pipeline.

The state file is intentionally independent from the official MIMIC split. It
must contain one row per ``study_id`` and a JSON/list column named
``clinical_states`` or ``radgraph_labels``.
"""

import ast
import json
from typing import Any, Dict


def parse_state_value(value: Any) -> Dict[str, Any]:
    """Parse a CSV cell into a JSON-serializable state dictionary."""
    if value is None:
        return {}
    if isinstance(value, float) and value != value:  # NaN
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {str(i): item for i, item in enumerate(value)}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, dict) else {str(i): x for i, x in enumerate(parsed)}
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return {"raw": text}


def normalize_state_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a state record while preserving arbitrary ontology fields."""
    if "study_id" not in record:
        raise ValueError("clinical state record requires study_id")
    state_column = "clinical_states" if "clinical_states" in record else "radgraph_labels"
    state = parse_state_value(record.get(state_column)) if state_column in record else {}
    return {"study_id": str(record["study_id"]), "clinical_states": state}
