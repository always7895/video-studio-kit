#!/usr/bin/env python3
"""Download or check the model files listed in setup/models.yaml.

    python setup/download_models.py --dry-run                 # HEAD check of URL and size
    python setup/download_models.py --models-dir D:\\models   # real download + SHA-256 check
    python setup/download_models.py --models-dir M --verify   # hash files already on disk

--models-dir defaults to the VSK_PATHS_MODELS_ROOT env var. Use --recipe to limit
the set. A Hugging Face token (HF_TOKEN) is sent only to huggingface.co if set.
Entries marked manual are never fetched. Downloads are written to <file>.part, hashed
while streaming, and renamed only when size and SHA-256 both match.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CHUNK = 8 * 1024 * 1024


def load_models(path: Path | None = None) -> list[dict]:
    doc = yaml.safe_load((path or HERE / "models.yaml").read_text(encoding="utf-8"))
    return doc["models"]


def target_path(models_dir: Path, model: dict) -> Path:
    name = model["name"]
    if "/" in name or "\\" in name or name in {"", ".", ".."} or ".." in name:
        raise ValueError(f"Unsafe model name: {name}")
    dest = Path(model["dest"])
    if dest.is_absolute() or ".." in dest.parts:
        raise ValueError(f"Unsafe destination: {dest}")
    return models_dir / dest / name


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    req = urllib.request.Request(url, method=method)
    token = os.environ.get("HF_TOKEN")
    if token and url.startswith("https://huggingface.co/"):
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "video-studio-kit-setup")
    return req


def head_size(url: str) -> int:
    with urllib.request.urlopen(_request(url, "HEAD"), timeout=60) as resp:
        length = resp.headers.get("X-Linked-Size") or resp.headers.get("Content-Length")
    if length is None:
        raise RuntimeError("no size header")
    return int(length)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(model: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    h, written = hashlib.sha256(), 0
    with urllib.request.urlopen(_request(model["url"]), timeout=120) as resp, part.open("wb") as fh:
        for block in iter(lambda: resp.read(CHUNK), b""):
            fh.write(block)
            h.update(block)
            written += len(block)
    if written != model["size"] or h.hexdigest() != model["sha256"]:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{model['name']}: size/SHA-256 mismatch (got {written} bytes, {h.hexdigest()})")
    part.replace(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="HEAD-check URLs and sizes; download nothing")
    ap.add_argument("--verify", action="store_true", help="hash existing files instead of downloading")
    ap.add_argument("--models-dir", default=os.environ.get("VSK_PATHS_MODELS_ROOT"))
    ap.add_argument("--recipe", action="append", help="only models used by this recipe (repeatable)")
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args(argv)

    models = load_models(Path(args.manifest) if args.manifest else None)
    if args.recipe:
        models = [m for m in models if m.get("recipe") in set(args.recipe)]
    failures = 0
    if args.dry_run:
        for m in models:
            if m.get("manual"):
                print(f"MANUAL  {m['name']}")
                continue
            try:
                size = head_size(m["url"])
                ok = size == m["size"]
                print(f"{'OK     ' if ok else 'SIZE?  '} {m['name']} remote={size} manifest={m['size']}")
                failures += 0 if ok else 1
            except Exception as exc:  # network/HTTP errors are reported, not hidden
                print(f"FAIL    {m['name']}: {exc}")
                failures += 1
        return 1 if failures else 0

    if not args.models_dir:
        ap.error("--models-dir (or VSK_PATHS_MODELS_ROOT) is required unless --dry-run")
    root = Path(args.models_dir)
    for m in models:
        out = target_path(root, m)
        if args.verify or m.get("manual"):
            if not out.is_file():
                print(f"MISSING {out}" + ("  (manual step)" if m.get("manual") else ""))
                failures += 1
            else:
                ok = out.stat().st_size == m["size"] and sha256_file(out) == m["sha256"]
                print(f"{'OK     ' if ok else 'BAD    '} {out}")
                failures += 0 if ok else 1
            continue
        if out.is_file() and out.stat().st_size == m["size"] and sha256_file(out) == m["sha256"]:
            print(f"HAVE    {out}")
            continue
        print(f"GET     {m['name']} ({m['size']} bytes)")
        try:
            download(m, out)
        except Exception as exc:
            print(f"FAIL    {m['name']}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
