"""Half-frame-rate (duplicated frame) detection for generated video."""
from __future__ import annotations

from typing import Sequence


def alt_ratio(diffs: Sequence[float]) -> float:
    """min(mean even-index diff, mean odd-index diff) / max of the two.

    A video whose every second frame is a near copy has one of the two means
    close to zero, so the ratio falls toward 0.
    """
    import numpy as np

    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 4:
        raise ValueError("Need at least 4 frame differences")
    a, b = float(d[0::2].mean()), float(d[1::2].mean())
    return min(a, b) / max(a, b, 1e-6)


def is_half_rate(diffs: Sequence[float], below: float = 0.3) -> bool:
    return alt_ratio(diffs) < below


def video_alt_ratio(path: str, size: tuple[int, int] = (270, 480)) -> tuple[int, float]:
    """Return (frame_count, alt_ratio) for a video file (needs opencv)."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    prev, diffs, n = None, [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                diffs.append(float(np.abs(gray - prev).mean()))
            prev = gray
    finally:
        cap.release()
    return n, alt_ratio(diffs)
