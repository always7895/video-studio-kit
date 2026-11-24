"""Threshold calibration for face backends from the user's own labelled pairs.

Pure Python (no numpy needed). A recommendation is only produced when enough pairs
exist; with fewer, it refuses. The output is advisory (disposition HOLD): a human
decides whether the threshold is acceptable for the footage.
"""
from __future__ import annotations

import math
from typing import Sequence

MIN_POSITIVE_PAIRS = 50
MIN_NEGATIVE_PAIRS = 300
Z95 = 1.959963984540054


class InsufficientSamples(ValueError):
    """Too few pairs to recommend a threshold."""


def wilson_interval(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes out of range")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    # The exact bounds at the extremes are 0 and 1; avoid floating-point residue.
    lo = 0.0 if successes == 0 else max(0.0, centre - half)
    hi = 1.0 if successes == total else min(1.0, centre + half)
    return lo, hi


def recommend_threshold(positive: Sequence[float], negative: Sequence[float], *,
                        max_fpr: float = 0.01,
                        min_positive: int = MIN_POSITIVE_PAIRS,
                        min_negative: int = MIN_NEGATIVE_PAIRS) -> dict:
    """Lowest threshold with false-positive rate <= ``max_fpr`` on the negative pairs.

    "Same person" means score >= threshold. Raises ``InsufficientSamples`` when
    there are fewer than ``min_positive`` positive or ``min_negative`` negative pairs.
    """
    pos = [float(x) for x in positive]
    neg = [float(x) for x in negative]
    if len(pos) < min_positive or len(neg) < min_negative:
        raise InsufficientSamples(
            f"need >= {min_positive} positive and >= {min_negative} negative pairs; "
            f"got {len(pos)} and {len(neg)}")
    if not all(math.isfinite(x) for x in pos + neg):
        raise ValueError("scores must be finite")
    if not 0.0 < max_fpr < 1.0:
        raise ValueError("max_fpr must be within (0, 1)")
    # Candidate thresholds: just above each negative score, plus the maximum.
    ordered = sorted(set(neg))
    candidates = [math.nextafter(x, math.inf) for x in ordered]
    candidates.append(max(ordered + pos) + 1e-9)
    chosen = None
    for t in sorted(candidates):
        fp = sum(1 for x in neg if x >= t)
        if fp / len(neg) <= max_fpr:
            chosen = t
            break
    assert chosen is not None  # the largest candidate always yields fp == 0
    # Report a 4-decimal threshold rounded UP, then recompute every statistic from the
    # reported value so that what the user copies is exactly what was measured.
    reported = math.ceil(chosen * 1e4) / 1e4
    fp = sum(1 for x in neg if x >= reported)
    while fp / len(neg) > max_fpr:  # guards against floating-point rounding at the boundary
        reported = round(reported + 1e-4, 4)
        fp = sum(1 for x in neg if x >= reported)
    tp = sum(1 for x in pos if x >= reported)
    return {
        "threshold": reported,
        "max_fpr": max_fpr,
        "positive_pairs": len(pos), "negative_pairs": len(neg),
        "fpr": fp / len(neg), "fpr_ci95": [round(v, 4) for v in wilson_interval(fp, len(neg))],
        "tpr": tp / len(pos), "tpr_ci95": [round(v, 4) for v in wilson_interval(tp, len(pos))],
        "disposition": "HOLD",
        "note": ("advisory only; a human reviewer decides whether to adopt this threshold. "
                 "The point was chosen on the same pairs it is measured on, and pairs of the "
                 "same people are not independent, so the intervals are optimistic: validate on a "
                 "separate set of images."),
    }


def scores_from_embeddings(by_person: dict) -> tuple[list[float], list[float]]:
    """Positive pairs (same person) and negative pairs (different people).

    ``by_person`` maps a person label to a list of embeddings, one per image.
    """
    from itertools import combinations

    from .identity import cosine

    pos: list[float] = []
    neg: list[float] = []
    for embs in by_person.values():
        pos.extend(cosine(a, b) for a, b in combinations(embs, 2))
    for (_, ea), (_, eb) in combinations(sorted(by_person.items()), 2):
        neg.extend(cosine(a, b) for a in ea for b in eb)
    return pos, neg


def embeddings_from_faces_dir(backend, faces_dir) -> tuple[dict, int]:
    """Layout: ``faces_dir/<person>/<image>``.

    Images with no face or with more than one face are skipped (a second face could belong
    to another person and would contaminate the pairs). Returns (embeddings, skipped_count).
    """
    from pathlib import Path

    import cv2
    import numpy as np

    out: dict = {}
    skipped = 0
    for person in sorted(p for p in Path(faces_dir).iterdir() if p.is_dir()):
        embs = []
        for img in sorted(person.iterdir()):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            frame = cv2.imdecode(np.fromfile(str(img), dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                skipped += 1
                continue
            found = backend.embeddings(frame)
            if len(found) == 1:
                embs.append(found[0])
            else:
                skipped += 1
        if len(embs) >= 2:
            out[person.name] = embs
    return out, skipped
