"""Config loader, graph builder and doctor tests. No ComfyUI, GPU or network."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from vsk.core import (
    ConfigError, build_graph, build_project, doctor, list_projects, list_recipes,
    load_local, load_project, load_qa, load_recipe, load_registry, resolve_path,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = ("moss", "ph3", "pika_h3")


class ConfigTests(unittest.TestCase):
    def test_recipe_catalog_and_offline_doctor(self):
        self.assertEqual(list_recipes(root=ROOT), RECIPES)
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            report = doctor(root=ROOT)
        self.assertEqual(report["network_calls"], 0)
        self.assertTrue(report["graph_sources_pinned"])
        self.assertEqual(report["recipes"], list(RECIPES))

    def test_exact_baseline_graph_equivalence_all_three(self):
        for name in RECIPES:
            with self.subTest(recipe=name):
                recipe = load_recipe(name, root=ROOT)
                source = ROOT / recipe["workflow"]["path"]
                raw = source.read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),
                                 recipe["workflow"]["sha256"])
                expected = json.loads(raw)
                actual = build_graph(name, root=ROOT)
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(set(actual), set(expected))

    def test_materialization_does_not_mutate_prior_graph(self):
        first = build_graph("pika_h3", root=ROOT)
        original = json.loads(json.dumps(first))
        second = build_graph("pika_h3", {"seed": 7654321}, root=ROOT)
        self.assertEqual(first, original)
        self.assertNotEqual(first, second)
        self.assertEqual(second["9"]["inputs"]["noise_seed"], 7654321)

    def test_pika_model_lora_and_dimensions_from_yaml(self):
        graph = build_graph("pika_h3",
                            {"model": "new-model.safetensors", "lora": "new-lora.safetensors",
                             "lora_strength": 0.75, "frames": 100,
                             "width": 640, "height": 1152}, root=ROOT)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "new-model.safetensors")
        self.assertEqual(graph["90"]["inputs"]["lora_name"], "new-lora.safetensors")
        self.assertEqual(graph["90"]["inputs"]["strength_model"], 0.75)
        self.assertEqual(graph["7"]["inputs"]["length"], 100)
        self.assertEqual((graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]),
                         (640, 1152))

    def test_moss_two_lora_strengths_and_steps_from_yaml(self):
        graph = build_graph("moss",
                            {"lora_deband": "deband-v2.safetensors",
                             "strength_sharp": 0.2, "steps": 40,
                             "batch_size": 2, "prompt": "Another fixture"},
                            root=ROOT)
        self.assertEqual(graph["100"]["inputs"]["lora_name"], "deband-v2.safetensors")
        self.assertEqual(graph["101"]["inputs"]["strength_model"], 0.2)
        self.assertEqual(graph["7"]["inputs"]["steps"], 40)
        self.assertEqual(graph["6"]["inputs"]["batch_size"], 2)
        self.assertEqual(graph["4"]["inputs"]["text"], "Another fixture")

    def test_ph3_resolution_and_sampler_from_yaml(self):
        graph = build_graph("ph3",
                            {"model": "p-h3-image_w4a8.safetensors",
                             "megapixels": 4, "sampler": "euler",
                             "prompt_upsampler_enabled": False}, root=ROOT)
        self.assertEqual(graph["22"]["inputs"]["unet_name"], "p-h3-image_w4a8.safetensors")
        self.assertEqual(graph["20"]["inputs"]["megapixels"], 4)
        self.assertFalse(graph["19"]["inputs"]["switch"])

    def test_sigma_steps_are_synchronized(self):
        with self.assertRaisesRegex(ConfigError, "Sigma schedule"):
            build_graph("pika_h3",
                        {"steps": {"count": 7, "sigmas":
                         "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"}},
                        root=ROOT)
        graph = build_graph("pika_h3",
                            {"steps": {"count": 2, "sigmas": "1.0, 0.5, 0.0"}}, root=ROOT)
        self.assertEqual(graph["91"]["inputs"]["sigmas"], "1.0, 0.5, 0.0")

    def test_unrecognized_and_type_mismatch_fail_closed(self):
        with self.assertRaisesRegex(ConfigError, "Unknown recipe parameter"):
            build_graph("moss", {"not_an_input": 5}, root=ROOT)
        with self.assertRaisesRegex(ConfigError, "Input type mismatch"):
            build_graph("ph3", {"seed": "zero"}, root=ROOT)
        with self.assertRaises(ConfigError):
            load_recipe("../tools", root=ROOT)

    def test_project_registry_qa_and_private_boundary(self):
        self.assertEqual(list_projects(root=ROOT), ("ad", "documentary", "explainer"))
        for name in list_projects(root=ROOT):
            with self.subTest(name=name):
                project = load_project(name, root=ROOT)
                self.assertEqual(project["root_key"], name)
                self.assertIn("character_masters", project["paths"])
                self.assertIn("output", project["paths"])
                self.assertEqual(project["qa"]["disposition"], "HOLD")
        self.assertIn("face_cosine_min", load_project("ad", root=ROOT)["qa"])
        self.assertIsNone(load_project("ad", root=ROOT)["qa"]["face_cosine_min"])
        self.assertEqual(len(load_registry("workflows", root=ROOT)["workflows"]), 3)
        self.assertEqual(load_qa(root=ROOT)["default_disposition"], "HOLD")
        self.assertIn("/local.yaml", (ROOT / "config/.gitignore").read_text())

    def test_every_example_project_builds_graphs_offline(self):
        expected = {"ad": {"hero_still", "hero_motion"}, "explainer": {"backdrop"},
                    "documentary": {"establishing", "establishing_motion"}}
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            for name, steps in expected.items():
                graphs = build_project(name, root=ROOT)
                self.assertEqual(set(graphs), steps)
                for graph in graphs.values():
                    self.assertTrue(graph)

    def test_project_overrides_reach_the_graph(self):
        graphs = build_project("ad", root=ROOT)
        self.assertEqual(graphs["hero_still"]["9"]["inputs"]["noise_seed"], 101)
        self.assertEqual(graphs["hero_motion"]["20"]["inputs"]["image"], "ad_hero_still.png")

    def test_resolve_path_prefers_env_then_local_and_rejects_placeholders(self):
        local = {"tools": {"ffmpeg": "/opt/ffmpeg"}, "paths": {"x": "<set-me>"}}
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VSK_TOOLS_FFMPEG", None)
            self.assertEqual(resolve_path("tools.ffmpeg", local=local), Path("/opt/ffmpeg"))
            self.assertIsNone(resolve_path("paths.x", local=local))
            self.assertIsNone(resolve_path("paths.missing", local=local))
            os.environ["VSK_TOOLS_FFMPEG"] = "/env/ffmpeg"
            self.assertEqual(resolve_path("tools.ffmpeg", local=local), Path("/env/ffmpeg"))

    def _scratch(self):
        return tempfile.TemporaryDirectory(prefix="vsk_test_")

    def _fixture_root(self, temp: str, name="moss"):
        root = Path(temp)
        recipe = load_recipe(name, root=ROOT)
        rel = Path(recipe["workflow"]["path"])
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / "config/recipes").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, root / rel)
        shutil.copy2(ROOT / f"config/recipes/{name}.yaml",
                     root / f"config/recipes/{name}.yaml")
        return root, recipe, rel

    def test_modified_json_is_refused_by_sha_pin(self):
        with self._scratch() as tmp:
            root, recipe, rel = self._fixture_root(tmp)
            with (root / rel).open("ab") as file:
                file.write(b" ")
            with self.assertRaisesRegex(ConfigError, "SHA256 mismatch"):
                load_recipe("moss", root=root)

    def test_replacement_workflow_requires_only_yaml_not_python_change(self):
        with self._scratch() as tmp:
            root, recipe, rel = self._fixture_root(tmp)
            replacement = root / "workflows/api/new_moss.api.json"
            replacement.parent.mkdir(parents=True, exist_ok=True)
            new_graph = json.loads((root / rel).read_text(encoding="utf-8"))
            new_graph["7"]["inputs"]["seed"] = 123
            replacement.write_text(json.dumps(new_graph, ensure_ascii=False),
                                   encoding="utf-8")
            recipe["workflow"]["path"] = "workflows/api/new_moss.api.json"
            recipe["workflow"]["sha256"] = hashlib.sha256(replacement.read_bytes()).hexdigest()
            recipe["defaults"]["seed"] = 123
            (root / "config/recipes/moss.yaml").write_text(
                yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False),
                encoding="utf-8")
            self.assertEqual(build_graph("moss", root=root), new_graph)

    def test_path_escape_rejected(self):
        with self._scratch() as tmp:
            root, recipe, _ = self._fixture_root(tmp)
            recipe["workflow"]["path"] = "../../untrusted.api.json"
            (root / "config/recipes/moss.yaml").write_text(
                yaml.safe_dump(recipe, allow_unicode=True), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_recipe("moss", root=root)

    def test_yaml_unsafe_tag_rejected(self):
        with self._scratch() as tmp:
            root, _, _ = self._fixture_root(tmp)
            (root / "config/recipes/moss.yaml").write_text(
                "!!python/object/apply:os.system ['echo never']\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_recipe("moss", root=root)


if __name__ == "__main__":
    unittest.main()
