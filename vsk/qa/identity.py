"""Face-identity screening with pluggable embedding backends (CPU).

``cosine``, ``score_identity`` and ``score_frames`` are pure numpy. Backends
(``insightface`` and ``yunet_sface``) load their libraries lazily and only on CPU.

There is deliberately no built-in threshold. Cosine scales differ between backends
and between footage types, so the threshold must come from the project YAML
(``qa.face_cosine_min``) or from ``vsk qa calibrate-face``. A missing threshold
raises ``ThresholdNotConfigured`` instead of silently using a default.

This module covers the interface and the threshold logic only. A low score is a flag
for a human, never an automatic verdict; every result carries ``disposition: HOLD``.
"""
from __future__ import annotations

import os
from typing import Mapping, Sequence


class ThresholdNotConfigured(ValueError):
    """No face threshold was supplied: run ``vsk qa calibrate-face`` and set it in the project YAML."""


def require_threshold(value) -> float:
    """Return a usable threshold or fail closed."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThresholdNotConfigured(
            "face threshold is not configured (qa.face_cosine_min is null); "
            "calibrate it with `vsk qa calibrate-face` first")
    if not -1.0 <= float(value) <= 1.0:
        raise ThresholdNotConfigured("face threshold must be within [-1, 1]")
    return float(value)


def threshold_from_project(project: Mapping, *, oblique: bool = False) -> float:
    """Read the frontal (default) or oblique threshold from a loaded project mapping."""
    qa = project.get("qa") if isinstance(project, Mapping) else None
    qa = qa if isinstance(qa, Mapping) else {}
    key = "face_cosine_min_oblique" if oblique else "face_cosine_min"
    return require_threshold(qa.get(key))


def cosine(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        raise ValueError("Zero-norm embedding")
    return float(np.dot(a, b) / denom)


def score_identity(references: Sequence, candidate, threshold: float | None = None) -> dict:
    """Best cosine against any reference embedding, with a screening flag.

    ``threshold`` has no default; ``None`` raises ``ThresholdNotConfigured``.
    """
    threshold = require_threshold(threshold)
    best = max(cosine(r, candidate) for r in references)
    return {"best_cosine": round(best, 4), "threshold": threshold,
            "flag_for_review": best < threshold, "disposition": "HOLD"}


def score_frames(references: Sequence, frame_embeddings: Mapping[int, object],
                 threshold: float | None = None) -> dict:
    """Per-frame scores plus the frames below threshold (frame -> embedding)."""
    threshold = require_threshold(threshold)
    scores = {f: score_identity(references, e, threshold) for f, e in frame_embeddings.items()}
    low = sorted(f for f, s in scores.items() if s["flag_for_review"])
    return {"frames": scores, "low_frames": low, "disposition": "HOLD"}


class FaceBackend:
    """Interface: turn a BGR image into one embedding per detected face, largest first."""

    name = "base"

    def embeddings(self, bgr_image) -> list:  # pragma: no cover - interface
        raise NotImplementedError


class InsightFaceBackend(FaceBackend):
    """insightface FaceAnalysis (ArcFace). Code is MIT; the buffalo_l weights are
    for non-commercial research use only (see NOTICE.md)."""

    name = "insightface"

    def __init__(self, model_root: str | None = None, name: str = "buffalo_l") -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        from insightface.app import FaceAnalysis

        from ..core import resolve_path

        root = model_root or (str(resolve_path("paths.insightface_root") or "") or None)
        kwargs = {"root": root} if root else {}
        self.app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"], **kwargs)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embeddings(self, bgr_image) -> list:
        faces = sorted(self.app.get(bgr_image),
                       key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)
        return [f.normed_embedding for f in faces]


ArcFaceEngine = InsightFaceBackend  # backwards-compatible name


class YuNetSFaceBackend(FaceBackend):
    """OpenCV Zoo YuNet detector plus SFace recognizer (128-d). Model files are manual
    downloads listed in setup/models.yaml; their locations come from
    ``paths.yunet_onnx`` and ``paths.sface_onnx`` (or VSK_PATHS_YUNET_ONNX / VSK_PATHS_SFACE_ONNX)."""

    name = "yunet_sface"

    def __init__(self, yunet_path: str | None = None, sface_path: str | None = None) -> None:
        import cv2

        from ..core import resolve_path

        yunet = yunet_path or str(resolve_path("paths.yunet_onnx") or "")
        sface = sface_path or str(resolve_path("paths.sface_onnx") or "")
        if not yunet or not sface or not os.path.isfile(yunet) or not os.path.isfile(sface):
            raise FileNotFoundError(
                "YuNet/SFace ONNX files not found: set paths.yunet_onnx and paths.sface_onnx")
        self.detector = cv2.FaceDetectorYN.create(yunet, "", (320, 320))
        self.recognizer = cv2.FaceRecognizerSF.create(sface, "")

    def embeddings(self, bgr_image) -> list:
        import numpy as np

        h, w = bgr_image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_image)
        if faces is None:
            return []
        order = sorted(faces, key=lambda f: float(f[2]) * float(f[3]), reverse=True)
        out = []
        for face in order:
            aligned = self.recognizer.alignCrop(bgr_image, face)
            feat = self.recognizer.feature(aligned).reshape(-1).astype(np.float64)
            out.append(feat / (np.linalg.norm(feat) or 1.0))
        return out


BACKENDS = {"insightface": InsightFaceBackend, "yunet_sface": YuNetSFaceBackend}


def make_backend(name: str, **kwargs) -> FaceBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown face backend: {name}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kwargs)
