"""Two-pass EBU R128 loudness mastering via ffmpeg ``loudnorm``.

``build_measure_cmd`` / ``build_apply_cmd`` only build argument lists; ``run``
executes them with the ffmpeg found through config/local.yaml
(``tools.ffmpeg``), VSK_TOOLS_FFMPEG, or PATH.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

TARGET_I, TARGET_TP, TARGET_LRA = -16.0, -1.5, 11.0


def ffmpeg_exe() -> str:
    from ..core import resolve_path

    configured = resolve_path("tools.ffmpeg")
    exe = str(configured) if configured else shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg not found: set tools.ffmpeg or put it on PATH")
    return exe


def _filter(i: float, tp: float, lra: float, measured: dict | None = None) -> str:
    base = f"loudnorm=I={i}:TP={tp}:LRA={lra}"
    if measured is None:
        return base + ":print_format=json"
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}:linear=true:print_format=summary")


def build_measure_cmd(src: str, ffmpeg: str = "ffmpeg", i: float = TARGET_I,
                      tp: float = TARGET_TP, lra: float = TARGET_LRA) -> list[str]:
    return [ffmpeg, "-hide_banner", "-nostdin", "-i", str(src), "-vn",
            "-af", _filter(i, tp, lra), "-f", "null", "-"]


def parse_measurement(stderr: str) -> dict:
    """Extract the trailing loudnorm JSON block from ffmpeg stderr."""
    blocks = re.findall(r"\{[^{}]*\}", stderr, flags=re.S)
    for text in reversed(blocks):
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if "input_i" in data:
            return data
    raise ValueError("No loudnorm measurement found in ffmpeg output")


def build_apply_cmd(src: str, dst: str, measured: dict, ffmpeg: str = "ffmpeg",
                    i: float = TARGET_I, tp: float = TARGET_TP, lra: float = TARGET_LRA,
                    audio_bitrate: str = "192k") -> list[str]:
    return [ffmpeg, "-hide_banner", "-nostdin", "-n", "-i", str(src), "-map", "0",
            "-c:v", "copy", "-af", _filter(i, tp, lra, measured),
            "-c:a", "aac", "-b:a", audio_bitrate, str(dst)]


def run(src: str, dst: str, cmd_runner=subprocess.run) -> dict:
    """Measure then apply. Refuses to overwrite ``dst``."""
    if Path(dst).exists():
        raise FileExistsError(dst)
    exe = ffmpeg_exe()
    measure = cmd_runner(build_measure_cmd(src, exe), capture_output=True, text=True)
    measured = parse_measurement(measure.stderr)
    cmd_runner(build_apply_cmd(src, dst, measured, exe), check=True)
    return measured
