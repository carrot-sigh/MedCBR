import numpy as np
import pyarrow as pa

from src.utils.hierarchy_screening_subset import create_screening_subset


def make_table(num_rows=100):
    c1_target, c1_mask = [], []
    c2_target, c2_mask = [], []
    c3_target, c3_mask = [], []
    for index in range(num_rows):
        c1 = np.zeros((2, 2), dtype=np.int8)
        c2 = np.zeros((2, 3), dtype=np.int8)
        c3 = np.zeros(2, dtype=np.int8)
        c1[index % 2, index % 2] = 1
        c2[index % 2, index % 3] = 1
        c3[index % 2] = 1
        c1_target.append(c1.tolist())
        c2_target.append(c2.tolist())
        c3_target.append(c3.tolist())
        c1_mask.append(np.ones_like(c1, dtype=bool).tolist())
        c2_mask.append(np.ones_like(c2, dtype=bool).tolist())
        c3_mask.append(np.ones_like(c3, dtype=bool).tolist())
    return pa.table({
        "c1_target": c1_target, "c1_mask": c1_mask,
        "c2_target": c2_target, "c2_mask": c2_mask,
        "c3_target": c3_target, "c3_mask": c3_mask,
    })


def test_screening_subset_is_deterministic_and_exact_size():
    table = make_table()
    first, first_report = create_screening_subset(table, 20, 17, 3)
    second, second_report = create_screening_subset(table, 20, 17, 3)
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 20
    assert first_report == second_report


def test_screening_subset_covers_every_positive_concept():
    indices, report = create_screening_subset(make_table(), 20, 17, 3)
    assert len(indices) == 20
    for counts in report["subset_positive_images"].values():
        assert all(count > 0 for count in counts)
