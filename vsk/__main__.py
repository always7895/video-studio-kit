"""Command line: ``vsk doctor``, ``vsk projects``, ``vsk build <project>``, ``vsk qa calibrate-face``.

Nothing here submits anything to ComfyUI. Precedence for configuration:
command-line option, then environment (VSK_LOCAL_CONFIG, VSK_PROJECTS_DIR, VSK_ROOT,
VSK_<KEY>), then the templates in the repository.

``doctor`` prints the resolved configuration paths. Do not paste that output into
public issues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .core import ConfigError, build_project, doctor, list_projects
from .core.config import _root


def _guard_out(out: str, root: str | None) -> Path:
    """Refuse an output directory that lies inside the repository."""
    target = Path(out).expanduser().resolve()
    repo = _root(root)
    if target == repo or repo in target.parents:
        raise ConfigError("--out must be outside the repository (it would be version-controlled)")
    return target


def _calibrate(args) -> int:
    from .qa.calibrate import (InsufficientSamples, embeddings_from_faces_dir,
                               recommend_threshold, scores_from_embeddings)

    if bool(args.scores) == bool(args.faces):
        raise ConfigError("give exactly one of --scores (JSON) or --faces (folder of person folders)")
    if args.scores:
        data = json.loads(Path(args.scores).read_text(encoding="utf-8"))
        pos, neg = data["positive"], data["negative"]
    else:
        from .qa.identity import make_backend

        pos, neg = scores_from_embeddings(
            embeddings_from_faces_dir(make_backend(args.backend), args.faces))
    try:
        result = recommend_threshold(pos, neg, max_fpr=args.max_fpr)
    except InsufficientSamples as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vsk", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="repository root (default: VSK_ROOT or checkout)")
    ap.add_argument("--local-config", default=None,
                    help="local config file (overrides VSK_LOCAL_CONFIG)")
    ap.add_argument("--projects-dir", default=None,
                    help="directory of project YAMLs (overrides VSK_PROJECTS_DIR)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="offline configuration check")
    sub.add_parser("projects", help="list project configs")
    b = sub.add_parser("build", help="build (never submit) the graphs of a project")
    b.add_argument("project")
    b.add_argument("--out", default=None,
                   help="write <step>.api.json files here (create-only, must be outside the repo)")
    q = sub.add_parser("qa", help="advisory QA tools")
    qsub = q.add_subparsers(dest="qa_cmd", required=True)
    c = qsub.add_parser("calibrate-face",
                        help="recommend a face threshold from your own positive and negative pairs")
    c.add_argument("--backend", default="insightface", choices=["insightface", "yunet_sface"])
    c.add_argument("--faces", default=None, help="folder with one sub-folder of images per person")
    c.add_argument("--scores", default=None, help='JSON {"positive": [...], "negative": [...]}')
    c.add_argument("--max-fpr", type=float, default=0.01)
    args = ap.parse_args(argv)
    if args.local_config:
        os.environ["VSK_LOCAL_CONFIG"] = str(Path(args.local_config).expanduser().resolve())
    if args.projects_dir:
        os.environ["VSK_PROJECTS_DIR"] = str(Path(args.projects_dir).expanduser().resolve())
    try:
        if args.cmd == "doctor":
            print(json.dumps(doctor(root=args.root), indent=2))
        elif args.cmd == "projects":
            print("\n".join(list_projects(root=args.root)))
        elif args.cmd == "qa":
            return _calibrate(args)
        else:
            out_dir = _guard_out(args.out, args.root) if args.out else None
            graphs = build_project(args.project, root=args.root)
            for name, graph in graphs.items():
                print(f"{name}: {len(graph)} nodes")
                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    target = out_dir / f"{args.project}_{name}.api.json"
                    with target.open("x", encoding="utf-8") as fh:
                        json.dump(graph, fh, indent=2)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
