"""Command line: ``python -m vsk doctor`` and ``python -m vsk build <project>``.

Both commands are offline and never submit anything to ComfyUI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import ConfigError, build_project, doctor, list_projects


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vsk", description=__doc__)
    ap.add_argument("--root", default=None, help="repository root (default: VSK_ROOT or checkout)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="offline configuration check")
    sub.add_parser("projects", help="list project configs")
    b = sub.add_parser("build", help="build (never submit) the graphs of a project")
    b.add_argument("project")
    b.add_argument("--out", default=None, help="write <step>.api.json files here (create-only)")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "doctor":
            print(json.dumps(doctor(root=args.root), indent=2))
        elif args.cmd == "projects":
            print("\n".join(list_projects(root=args.root)))
        else:
            graphs = build_project(args.project, root=args.root)
            for name, graph in graphs.items():
                print(f"{name}: {len(graph)} nodes")
                if args.out:
                    target = Path(args.out) / f"{args.project}_{name}.api.json"
                    with target.open("x", encoding="utf-8") as fh:
                        json.dump(graph, fh, indent=2)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
