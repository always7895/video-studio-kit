#!/usr/bin/env python3
"""Install ComfyUI and the public custom nodes at the commits in setup/pins.yaml.

    python setup/install_comfyui.py --dest /path/to/ComfyUI

Only git is used. Python dependencies are NOT installed here; follow the README
(``pip install -r <dest>/requirements.txt`` in your own venv, with the torch build
that matches your GPU). An existing destination is never overwritten.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def clone_at(url: str, commit: str, dest: Path) -> None:
    """Fetch exactly one commit (shallow) and check it out detached."""
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite")
    dest.mkdir(parents=True)
    git("init", "-q", cwd=dest)
    git("remote", "add", "origin", url, cwd=dest)
    git("fetch", "-q", "--depth", "1", "origin", commit, cwd=dest)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != commit:
        raise RuntimeError(f"{dest.name}: expected {commit}, got {head}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="new ComfyUI directory (must not exist)")
    ap.add_argument("--pins", default=str(HERE / "pins.yaml"))
    ap.add_argument("--no-nodes", action="store_true", help="install ComfyUI only")
    args = ap.parse_args(argv)

    pins = yaml.safe_load(Path(args.pins).read_text(encoding="utf-8"))
    dest = Path(args.dest).resolve()
    comfy = pins["comfyui"]
    print(f"ComfyUI {comfy['tag']} at {comfy['commit']} -> {dest}")
    clone_at(comfy["url"], comfy["commit"], dest)
    if not args.no_nodes:
        for node in pins["custom_nodes"]:
            print(f"node {node['name']} at {node['commit']}")
            clone_at(node["url"], node["commit"], dest / "custom_nodes" / node["name"])
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
