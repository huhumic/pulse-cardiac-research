import numpy as np
from pulse_repro.metrics import measure_binary

def test_metrics_perfect_and_shifted():
    mask = np.zeros((9, 10, 11), dtype=bool)
    mask[2:6, 3:8, 4:9] = True
    m = measure_binary(mask, mask, (1, 1, 2.5))
    assert m["status"] == "valid"
    assert m["dice"] == 1.0 and m["hd95_mm"] == 0
    moved = np.roll(mask, 1, axis=2)
    n = measure_binary(mask, moved, (1, 1, 2.5))
    assert n["dice"] < 1 and n["hd95_mm"] > 0

def test_empty_structure_propagates_flags():
    mask = np.zeros((4, 4, 4), dtype=bool)
    full = mask.copy()
    full[1, 1, 1] = True
    m = measure_binary(full, mask, (1, 1, 1))
    assert m["dice"] == 0 and m["status"] == "one_empty"
    assert np.isnan(m["hd95_mm"])
    assert measure_binary(mask, mask, (1, 1, 1))["status"] == "both_empty"
