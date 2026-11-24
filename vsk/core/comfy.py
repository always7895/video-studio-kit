"""Bounded, read-only ComfyUI API client (stdlib only).

Loopback endpoints only; no retries and no submissions. A caller that submits
work must first check ``require_idle`` and hold its own device lease.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, NamedTuple


class ComfyError(RuntimeError):
    pass


class Queue(NamedTuple):
    running: int
    pending: int

    def _is_idle(self) -> bool:
        return self.running == 0 and self.pending == 0

    idle = property(_is_idle)


def _object_response(response: Any) -> dict:
    try:
        data = json.load(response)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ComfyError("Malformed ComfyUI JSON response") from exc
    if not isinstance(data, dict):
        raise ComfyError("ComfyUI response must be a JSON object")
    return data


class ComfyClient:
    """Loopback-only transport; opener can be a strictly sandboxed fake."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188",
                 opener: Callable[..., Any] | None = None,
                 sandbox_only: bool = False) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme != "http" or (parsed.hostname not in {"127.0.0.1", "localhost"}
                 and not (sandbox_only and parsed.hostname == "mock.invalid" and opener is not None))
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in {"", "/"}
                or (not parsed.port and not (sandbox_only and parsed.hostname == "mock.invalid"))):
            raise ComfyError("Only explicit loopback HTTP ComfyUI endpoints are permitted")
        self.base_url = base_url.rstrip("/")
        self.opener = opener or urllib.request.urlopen

    def _get(self, route: str, timeout: int = 15) -> dict:
        if not route.startswith("/") or route.startswith("//") or ".." in route:
            raise ComfyError("Unsafe ComfyUI GET route")
        request = urllib.request.Request(self.base_url + route, method="GET")
        with self.opener(request, timeout=timeout) as response:
            return _object_response(response)

    def queue_snapshot(self) -> dict:
        response = self._get("/queue")
        validate_queue(response)
        return response

    def history(self, *, max_items: int = 100000) -> dict:
        if not 1 <= max_items <= 100000:
            raise ComfyError("Invalid history bound")
        items = self._get(f"/history?max_items={max_items}", timeout=45)
        if len(items) >= max_items:
            raise ComfyError("Potentially truncated ComfyUI history")
        return items

    def prompt_history(self, prompt_id: str) -> dict:
        if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prompt_id):
            raise ComfyError("Invalid prompt ID")
        return self._get("/history/" + prompt_id, timeout=20)

    def output_bytes(self, *, filename: str, subfolder: str = "", storage: str = "output") -> bytes:
        if storage not in {"output", "temp"} or not isinstance(filename, str):
            raise ComfyError("Invalid output query")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,255}", filename):
            raise ComfyError("Unsafe output basename")
        if subfolder and (subfolder.startswith(("/", "\\")) or
                          any(p in {"", ".", ".."} for p in subfolder.replace("\\", "/").split("/"))):
            raise ComfyError("Unsafe output subfolder")
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": storage})
        request = urllib.request.Request(self.base_url + "/view?" + params, method="GET")
        with self.opener(request, timeout=30) as response:
            data = response.read(64 * 1024 * 1024 + 1)
        if len(data) > 64 * 1024 * 1024:
            raise ComfyError("Output exceeds 64 MiB")
        return data


def validate_queue(queue: dict) -> Queue:
    if not isinstance(queue, dict) or not isinstance(queue.get("queue_running"), list) or not isinstance(
            queue.get("queue_pending"), list):
        raise ComfyError("Malformed ComfyUI queue")
    return Queue(len(queue["queue_running"]), len(queue["queue_pending"]))


def require_idle(queue: dict) -> Queue:
    """Raise ComfyError unless the queue is 0 running / 0 pending."""
    snapshot = validate_queue(queue)
    if not snapshot.idle:
        raise ComfyError(f"ComfyUI queue not idle: {snapshot.running}/{snapshot.pending}")
    return snapshot
