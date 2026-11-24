"""Background optical-flow screening: motion outside the subject's boxes."""
from __future__ import annotations

from typing import Iterable, Sequence

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


def background_mask(shape: tuple[int, int], boxes: Iterable[Box], margin: float = 0.25):
    """Boolean mask, True on background; each box is grown by ``margin``."""
    import numpy as np

    h, w = shape
    mask = np.ones((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
        mask[max(0, y0 - my):min(h, y1 + my), max(0, x0 - mx):min(w, x1 + mx)] = False
    return mask


def background_flow(prev_gray, gray, boxes: Iterable[Box] = ()) -> float:
    """Mean Farneback flow magnitude (pixels) over the background region."""
    import cv2
    import numpy as np

    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    mask = background_mask(gray.shape[:2], boxes)
    return float(mag[mask].mean()) if mask.any() else 0.0


def sequence_flow(frames: Sequence, boxes_per_frame: Sequence[Iterable[Box]] | None = None) -> list[float]:
    out = []
    for i in range(1, len(frames)):
        boxes = boxes_per_frame[i] if boxes_per_frame else ()
        out.append(background_flow(frames[i - 1], frames[i], boxes))
    return out
