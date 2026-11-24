"""Calibration maths, overlay precedence and the --out guard. Synthetic data only."""
import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vsk.__main__ import main
from vsk.core import ConfigError, doctor, list_projects, load_project, resolve_path
from vsk.qa.calibrate import InsufficientSamples, recommend_threshold, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
PROJECT = """schema_version: 1
id: {name}
name: overlay example
root_key: {name}
paths: {{character_masters: a, output: b}}
qa: {{profile: standard, face_cosine_min: 0.5, disposition: HOLD}}
steps:
  - name: s
    recipe: moss
    overrides: {{prompt: "x", filename_prefix: vsk/o/s}}
"""


def scores(seed=1):
    rng = random.Random(seed)
    return ([rng.gauss(0.7, 0.08) for _ in range(60)], [rng.gauss(0.1, 0.1) for _ in range(400)])


class Calibrate(unittest.TestCase):
    def test_recommendation_and_intervals(self):
        pos, neg = scores()
        out = recommend_threshold(pos, neg)
        self.assertLessEqual(out["fpr"], 0.01)
        self.assertEqual(out["disposition"], "HOLD")
        lo, hi = out["tpr_ci95"]
        self.assertLessEqual(lo, out["tpr"])
        self.assertGreaterEqual(hi, out["tpr"])

    def test_refuses_small_samples(self):
        pos, neg = scores()
        with self.assertRaises(InsufficientSamples):
            recommend_threshold(pos[:49], neg)
        with self.assertRaises(InsufficientSamples):
            recommend_threshold(pos, neg[:299])

    def test_wilson_known_value(self):
        lo, hi = wilson_interval(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0370, places=3)

    def test_cli_scores_and_refusal(self):
        pos, neg = scores()
        with tempfile.TemporaryDirectory() as d:
            ok, small = Path(d, "ok.json"), Path(d, "small.json")
            ok.write_text(json.dumps({"positive": pos, "negative": neg}))
            small.write_text(json.dumps({"positive": pos[:5], "negative": neg[:5]}))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["qa", "calibrate-face", "--scores", str(ok)]), 0)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["qa", "calibrate-face", "--scores", str(small)]), 3)


class Overlay(unittest.TestCase):
    def test_projects_dir_overrides_and_doctor_reports(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "ad.yaml").write_text(PROJECT.format(name="ad"))
            Path(d, "mine.yaml").write_text(PROJECT.format(name="mine"))
            with patch.dict(os.environ, {"VSK_PROJECTS_DIR": d}):
                self.assertEqual(load_project("ad", root=ROOT)["qa"]["face_cosine_min"], 0.5)
                self.assertIn("mine", list_projects(root=ROOT))
                rep = doctor(root=ROOT)
                self.assertEqual(rep["project_sources"]["ad"], "overlay")
                self.assertEqual(rep["project_sources"]["explainer"], "repo")
        self.assertIsNone(load_project("ad", root=ROOT)["qa"]["face_cosine_min"])

    def test_local_config_env_wins_and_missing_fails(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d, "local.yaml")
            cfg.write_text("schema_version: 1\ntools:\n  ffmpeg: " + str(Path(d, "ff")).replace("\\", "/") + "\n")
            with patch.dict(os.environ, {"VSK_LOCAL_CONFIG": str(cfg)}):
                self.assertEqual(resolve_path("tools.ffmpeg", root=ROOT), Path(d, "ff"))
                self.assertEqual(doctor(root=ROOT)["local_config"]["source"], "env VSK_LOCAL_CONFIG")
            with patch.dict(os.environ, {"VSK_LOCAL_CONFIG": str(Path(d, "missing.yaml"))}):
                with self.assertRaises(ConfigError):
                    doctor(root=ROOT)

    def test_build_out_refuses_repo_inside(self):
        with redirect_stderr(io.StringIO()):
            rc = main(["--root", str(ROOT), "build", "explainer", "--out", str(ROOT / "tmp_out")])
        self.assertEqual(rc, 2)
        self.assertFalse((ROOT / "tmp_out").exists())
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(ROOT), "build", "explainer", "--out", d]), 0)
            self.assertTrue(Path(d, "explainer_backdrop.api.json").is_file())


if __name__ == "__main__":
    unittest.main()
