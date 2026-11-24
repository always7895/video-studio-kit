"""M1 configuration loader and CPU-only ComfyUI graph materializer.

This module never contacts ComfyUI, starts a service, or submits a graph.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Invalid, missing, or untrusted local configuration."""


_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENCODINGS = {"literal", "sigma_schedule"}


def _root(root: str | Path | None) -> Path:
    """Repository root: explicit argument, then VSK_ROOT, then the checkout."""
    if root is None and os.environ.get("VSK_ROOT"):
        root = os.environ["VSK_ROOT"]
    return (Path(root) if root is not None else Path(__file__).resolve().parents[2]).resolve()


def _safe_file(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ConfigError("A repository-relative path is required")
    # Check each literal filesystem component before resolve. A resolved
    # symlink/junction no longer reports the link that was traversed.
    raw = root / relative
    current = Path(raw.anchor)
    for part in raw.parts[1:] if raw.anchor else raw.parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ConfigError("Linked configuration/workflow path forbidden")
    candidate = raw.resolve()
    if not candidate.is_relative_to(root):
        raise ConfigError("Path escapes the repository")
    if not candidate.is_file():
        raise ConfigError(f"Missing configuration/source file: {relative}")
    return candidate


def _name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ConfigError("Invalid configuration name")
    return name


def _yaml(relative: str, root: Path) -> dict[str, Any]:
    path = _safe_file(root, relative)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load YAML: {relative}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"YAML document must be a mapping: {relative}")
    if type(doc.get("schema_version")) is not int or doc["schema_version"] != 1:
        raise ConfigError(f"Unsupported schema_version in {relative}")
    return doc


def load_registry(name: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """Return a copy of one committed models/workflows registry file."""
    if name not in {"models", "workflows"}:
        raise ConfigError("Registry must be models or workflows")
    return _yaml(f"config/registry/{name}.yaml", _root(root))


def load_qa(*, root: str | Path | None = None) -> dict[str, Any]:
    return _yaml("config/qa.yaml", _root(root))


def load_project(name: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """Project roots are symbolic; their actual values belong in ignored local.yaml."""
    name = _name(name)
    doc = _yaml(f"config/projects/{name}.yaml", _root(root))
    if doc.get("id") != name or not isinstance(doc.get("root_key"), str):
        raise ConfigError(f"Invalid project id/root_key: {name}")
    if not isinstance(doc.get("paths"), dict) or not isinstance(doc.get("qa"), dict):
        raise ConfigError(f"Invalid project paths/qa: {name}")
    return doc


def load_local(*, root: str | Path | None = None) -> dict[str, Any]:
    """Only reads local.yaml. Never creates it or falls back to example credentials."""
    base = _root(root)
    if not (base / "config/local.yaml").is_file():
        return {}
    return _yaml("config/local.yaml", base)


def _load_graph(recipe: Mapping[str, Any], root: Path) -> dict[str, Any]:
    workflow = recipe.get("workflow")
    if not isinstance(workflow, dict) or not isinstance(workflow.get("path"), str):
        raise ConfigError("Missing workflow.path")
    digest = workflow.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ConfigError("Invalid workflow.sha256")
    data = _safe_file(root, workflow["path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ConfigError("Workflow SHA256 mismatch; update the recipe pin after review")
    try:
        graph = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Workflow is not valid UTF-8 JSON") from exc
    if not isinstance(graph, dict) or not graph:
        raise ConfigError("Workflow must be a nonempty API node mapping")
    for node in graph.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ConfigError("Workflow contains a node without inputs")
    return graph


def _encode(value: Any, encoding: str) -> Any:
    if encoding == "literal":
        return copy.deepcopy(value)
    if encoding != "sigma_schedule" or not isinstance(value, dict):
        raise ConfigError("steps require {count, sigmas} with sigma_schedule encoding")
    count, sigmas = value.get("count"), value.get("sigmas")
    if type(count) is not int or not 1 <= count <= 1000 or not isinstance(sigmas, str):
        raise ConfigError("Invalid sigma step count or schedule")
    try:
        values = [float(piece.strip()) for piece in sigmas.split(",")]
    except ValueError as exc:
        raise ConfigError("Sigma schedule contains non-numeric values") from exc
    if (len(values) != count + 1 or not all(math.isfinite(v) for v in values)
            or values[0] != 1.0 or values[-1] != 0.0
            or any(a < b or b < 0.0 for a, b in zip(values, values[1:]))):
        raise ConfigError("Sigma schedule does not match the step count or monotonic bounds")
    return sigmas


def _apply(graph: dict[str, Any], parameters: Mapping[str, Any],
           values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        binding = parameters.get(key)
        if not isinstance(binding, dict):
            raise ConfigError(f"Unknown recipe parameter: {key}")
        node_id, input_name = binding.get("node"), binding.get("input")
        if not isinstance(node_id, str) or not isinstance(input_name, str):
            raise ConfigError(f"Invalid node/input binding: {key}")
        try:
            inputs = graph[node_id]["inputs"]
            current = inputs[input_name]
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Binding target absent: {key} at {node_id}.{input_name}") from exc
        encoding = binding.get("encoding", "literal")
        if encoding not in _ENCODINGS:
            raise ConfigError(f"Unsupported encoding for {key}")
        replacement = _encode(value, encoding)
        # Reject accidental changes of ComfyUI input type, including bool vs int.
        if type(current) is not type(replacement):
            raise ConfigError(f"Input type mismatch: {key}")
        inputs[input_name] = replacement


def load_recipe(name: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """Load a pinned recipe; fail closed on missing graph nodes or stale SHA."""
    name = _name(name)
    base = _root(root)
    doc = _yaml(f"config/recipes/{name}.yaml", base)
    if doc.get("id") != name or doc.get("kind") not in {"still", "video"}:
        raise ConfigError("Invalid recipe identity/kind")
    params, defaults = doc.get("parameters"), doc.get("defaults")
    if not isinstance(params, dict) or not params or not isinstance(defaults, dict):
        raise ConfigError("Recipe must declare parameters and defaults mappings")
    graph = _load_graph(doc, base)
    # Validate *all* bindings, including those without a default such as prompt.
    for field in params:
        if not isinstance(field, str):
            raise ConfigError("Recipe parameter keys must be strings")
        spec = params[field]
        if not isinstance(spec, dict) or spec.get("encoding", "literal") not in _ENCODINGS:
            raise ConfigError(f"Invalid binding: {field}")
        node, input_name = spec.get("node"), spec.get("input")
        if not isinstance(node, str) or not isinstance(input_name, str):
            raise ConfigError(f"Invalid binding target: {field}")
        if node not in graph or input_name not in graph[node]["inputs"]:
            raise ConfigError(f"Missing graph input for {field}: {node}.{input_name}")
    _apply(graph, params, defaults)
    return doc


def build_graph(name: str, overrides: Mapping[str, Any] | None = None,
                *, root: str | Path | None = None) -> dict[str, Any]:
    """Return a new, never-submitted ComfyUI API graph from YAML and pinned JSON.

    Unmapped node fields stay byte-equivalent in meaning to the source graph.
    Overrides use the same parameter names as the recipe; a sigma step change
    must include a complete, length-matched schedule in its steps object.
    """
    base = _root(root)
    recipe = load_recipe(name, root=base)
    graph = _load_graph(recipe, base)
    _apply(graph, recipe["parameters"], recipe["defaults"])
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigError("Overrides must be a mapping")
        _apply(graph, recipe["parameters"], overrides)
    return graph


def list_recipes(*, root: str | Path | None = None) -> tuple[str, ...]:
    """Only recipe names; no GPU inspection or filesystem writes."""
    path = _root(root) / "config/recipes"
    return tuple(sorted(p.stem for p in path.glob("*.yaml") if _NAME.fullmatch(p.stem)))


def list_projects(*, root: str | Path | None = None) -> tuple[str, ...]:
    path = _root(root) / "config/projects"
    return tuple(sorted(p.stem for p in path.glob("*.yaml") if _NAME.fullmatch(p.stem)))


def resolve_path(key: str, *, root: str | Path | None = None,
                 local: Mapping[str, Any] | None = None) -> Path | None:
    """Resolve a machine path from env var VSK_<KEY> or config/local.yaml.

    ``key`` is dotted, e.g. ``tools.ffmpeg`` -> env ``VSK_TOOLS_FFMPEG`` or
    ``local["tools"]["ffmpeg"]``. Returns None when unset or still a placeholder
    (``<...>``). Nothing in the repository hardcodes a machine path.
    """
    env = os.environ.get("VSK_" + re.sub(r"[^A-Za-z0-9]+", "_", key).upper())
    value: Any = env
    if not value:
        node: Any = local if local is not None else load_local(root=root)
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        value = node
    if not isinstance(value, str) or not value or value.startswith("<"):
        return None
    return Path(value).expanduser()


def build_project(name: str, *, root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Build (never submit) every step graph declared by a project YAML.

    Each ``steps`` entry is ``{name, recipe, overrides}``; the result maps the
    step name to a fresh ComfyUI API graph.
    """
    base = _root(root)
    project = load_project(name, root=base)
    steps = project.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ConfigError(f"Project declares no steps: {name}")
    graphs: dict[str, dict[str, Any]] = {}
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str)                 or not isinstance(step.get("recipe"), str):
            raise ConfigError(f"Invalid step in project {name}")
        if step["name"] in graphs:
            raise ConfigError(f"Duplicate step name in project {name}: {step['name']}")
        graphs[step["name"]] = build_graph(step["recipe"], step.get("overrides"), root=base)
    return graphs


def doctor(*, root: str | Path | None = None) -> dict[str, Any]:
    """Offline/read-only configuration diagnosis; never calls a service."""
    base = _root(root)
    recipes = list_recipes(root=base)
    for name in recipes:
        load_recipe(name, root=base)
    projects = list_projects(root=base)
    for project in projects:
        load_project(project, root=base)
        build_project(project, root=base)
    load_registry("models", root=base)
    load_registry("workflows", root=base)
    load_qa(root=base)
    return {"recipes": list(recipes), "projects": list(projects),
            "local_present": (base / "config/local.yaml").is_file(),
            "graph_sources_pinned": True, "network_calls": 0}


class RecipeSnapshot:
    """Immutable-by-convention source buffers, parsed once, with fresh graph copies."""

    def __init__(self, name: str, recipe: dict[str, Any],
                 graph: dict[str, Any], recipe_sha: str,
                 workflow_sha: str, workflow_path: str) -> None:
        self.name = name
        self._recipe = copy.deepcopy(recipe)
        self._graph = copy.deepcopy(graph)
        self.recipe_sha256 = recipe_sha
        self.workflow_sha256 = workflow_sha
        self.workflow_path = workflow_path

    def _recipe_copy(self) -> dict[str, Any]:
        return copy.deepcopy(self._recipe)

    recipe = property(_recipe_copy)

    def materialize(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        graph = copy.deepcopy(self._graph)
        _apply(graph, self._recipe["parameters"], self._recipe["defaults"])
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise ConfigError("Overrides must be a mapping")
            _apply(graph, self._recipe["parameters"], overrides)
        return graph


def load_recipe_snapshot(name: str, *, root: str | Path | None = None) -> RecipeSnapshot:
    """Read each YAML and workflow JSON exactly once, verify their actual bytes.

    Parsed graph materialization never reopens either source.
    """
    base, name = _root(root), _name(name)
    recipe_raw = _safe_file(base, f"config/recipes/{name}.yaml").read_bytes()
    try:
        doc = yaml.safe_load(recipe_raw.decode("utf-8-sig"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError("Invalid pinned recipe YAML") from exc
    if (not isinstance(doc, dict) or doc.get("schema_version") != 1 or
            doc.get("id") != name or doc.get("kind") not in {"still", "video"} or
            not isinstance(doc.get("defaults"), dict) or
            not isinstance(doc.get("parameters"), dict) or
            not isinstance(doc.get("workflow"), dict)):
        raise ConfigError("Invalid snapshot recipe")
    workflow_path = doc["workflow"].get("path")
    pinned_sha = doc["workflow"].get("sha256")
    if not isinstance(pinned_sha, str) or re.fullmatch("[0-9a-f]{64}", pinned_sha) is None:
        raise ConfigError("Invalid snapshot workflow SHA")
    workflow_raw = _safe_file(base, workflow_path).read_bytes()
    observed = hashlib.sha256(workflow_raw).hexdigest()
    if observed != pinned_sha:
        raise ConfigError("Snapshot workflow SHA256 mismatch")
    try:
        graph = json.loads(workflow_raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("Snapshot workflow JSON invalid") from exc
    if not isinstance(graph, dict) or not graph:
        raise ConfigError("Snapshot graph must be a nonempty mapping")
    for node in graph.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise ConfigError("Snapshot node invalid")
    for key, binding in doc["parameters"].items():
        if (not isinstance(key, str) or not isinstance(binding, dict) or
                binding.get("encoding", "literal") not in _ENCODINGS):
            raise ConfigError("Snapshot binding invalid")
        node, field = binding.get("node"), binding.get("input")
        if not isinstance(node, str) or not isinstance(field, str):
            raise ConfigError("Snapshot binding target invalid")
        if node not in graph or field not in graph[node]["inputs"]:
            raise ConfigError("Snapshot binding target missing")
    _apply(copy.deepcopy(graph), doc["parameters"], doc["defaults"])
    return RecipeSnapshot(
        name, doc, graph, hashlib.sha256(recipe_raw).hexdigest(),
        observed, workflow_path)
