"""Face-identity screening with ArcFace embeddings (CPU).

``cosine`` and ``score_identity`` are pure numpy. ``ArcFaceEngine`` loads
insightface lazily and only on CPU; its model directory comes from
``paths.insightface_root`` in config/local.yaml or VSK_PATHS_INSIGHTFACE_ROOT.
A low score is a flag for a human, never an automatic verdict.
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence


def cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("Zero-norm embedding")
    return float(np.dot(a, b) / denom)


def score_identity(references: Sequence, candidate, threshold: float = 0.65) -> dict:
    """Best cosine against any reference embedding, with a screening flag."""
    best = max(cosine(r, candidate) for r in references)
    return {"best_cosine": round(best, 4), "threshold": threshold,
            "flag_for_review": best < threshold, "disposition": "HOLD"}


def score_frames(references: Sequence, frame_embeddings: Mapping[int, object],
                 threshold: float = 0.65) -> dict:
    """Per-frame scores plus the frames below threshold (frame -> embedding)."""
    scores = {f: score_identity(references, e, threshold) for f, e in frame_embeddings.items()}
    low = sorted(f for f, s in scores.items() if s["flag_for_review"])
    return {"frames": scores, "low_frames": low, "disposition": "HOLD"}


class ArcFaceEngine:
    """Thin CPU wrapper around insightface FaceAnalysis (``qa`` extra)."""

    def __init__(self, model_root: str | None = None, name: str = "buffalo_l") -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        from insightface.app import FaceAnalysis

        from ..core import resolve_path

        root = model_root or (str(resolve_path("paths.insightface_root") or "") or None)
        kwargs = {"root": root} if root else {}
        self.app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"], **kwargs)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embeddings(self, bgr_image) -> list:
        """One normalised embedding per detected face, largest face first."""
        faces = sorted(self.app.get(bgr_image),
                       key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)
        return [f.normed_embedding for f in faces]
