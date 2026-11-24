"""Static checks of setup/models.yaml and setup/pins.yaml (no network)."""
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "setup"))


class SetupTests(unittest.TestCase):
    def test_models_manifest_shape_and_recipe_coverage(self):
        doc = yaml.safe_load((ROOT / "setup/models.yaml").read_text(encoding="utf-8"))
        names = set()
        for m in doc["models"]:
            names.add(m["name"])
            self.assertTrue(m["license"])
            self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$")
            self.assertIsInstance(m["size"], int)
            if m.get("manual"):
                continue
            self.assertRegex(m["url"], r"^https://huggingface\.co/.+/resolve/[0-9a-f]{40}/")
        registry = yaml.safe_load((ROOT / "config/registry/models.yaml").read_text(encoding="utf-8"))
        for entry in registry["models"].values():
            for key in ("diffusion", "text_encoder", "lora"):
                if key in entry:
                    self.assertIn(entry[key], names)

    def test_pins_are_full_commits(self):
        pins = yaml.safe_load((ROOT / "setup/pins.yaml").read_text(encoding="utf-8"))
        self.assertRegex(pins["comfyui"]["commit"], r"^[0-9a-f]{40}$")
        for node in pins["custom_nodes"]:
            self.assertRegex(node["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(node["url"].startswith("https://github.com/"))

    def test_download_helpers(self):
        import download_models as dm

        self.assertEqual(dm.target_path(Path("m"), {"name": "a.safetensors", "dest": "vae"}),
                         Path("m") / "vae" / "a.safetensors")
        with self.assertRaises(ValueError):
            dm.target_path(Path("m"), {"name": "../evil", "dest": "vae"})
