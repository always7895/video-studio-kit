"""Cut, freeze and in-shot jump screening from grayscale frame sequences."""
from __future__ import annotations

from typing import Sequence


def frame_diffs(frames: Sequence) -> list[float]:
    """Mean absolute difference between consecutive grayscale frames."""
    import numpy as np

    return [float(np.abs(np.asarray(frames[i], dtype=np.float32)
                         - np.asarray(frames[i - 1], dtype=np.float32)).mean())
            for i in range(1, len(frames))]


def detect_cuts(diffs: Sequence[float], ratio: float = 6.0, floor: float = 12.0) -> list[int]:
    """Indices i where diffs[i] (frame i -> i+1) is a hard cut.

    A cut exceeds both ``floor`` and ``ratio`` x the median of its neighbours.
    """
    import numpy as np

    d = np.asarray(diffs, dtype=np.float64)
    cuts = []
    for i, v in enumerate(d):
        if v <= floor:
            continue
        lo, hi = max(0, i - 6), min(len(d), i + 7)
        neighbours = np.delete(d[lo:hi], i - lo)
        med = float(np.median(neighbours)) if neighbours.size else 0.0
        if v > ratio * max(med, 1e-6):
            cuts.append(i)
    return cuts


def freeze_runs(diffs: Sequence[float], eps: float = 0.3, min_len: int = 3) -> list[tuple[int, int]]:
    """Runs [start, end) of >= min_len consecutive near-zero differences."""
    runs, start = [], None
    for i, v in enumerate(list(diffs) + [float("inf")]):
        if v < eps:
            start = i if start is None else start
        else:
            if start is not None and i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def in_shot_jumps(diffs: Sequence[float], cuts: Sequence[int], k: float = 4.0,
                  floor: float = 12.0) -> list[int]:
    """Large differences that are not declared cuts (suspicious in-shot jumps)."""
    import numpy as np

    d = np.asarray(diffs, dtype=np.float64)
    cutset = set(cuts)
    keep = np.array([i not in cutset for i in range(len(d))])
    if not keep.any():
        return []
    med = float(np.median(d[keep]))
    return [i for i in range(len(d)) if keep[i] and d[i] > floor and d[i] > k * max(med, 1e-6)]
