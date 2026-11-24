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
                self.assertEqual(doctor(root=ROOT)["local_config"]["source"],
                                 "override (--local-config or VSK_LOCAL_CONFIG)")
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


class Review2Fixes(unittest.TestCase):
    def test_threshold_rounded_up_never_admits_extra_negative(self):
        # 4th-highest negative sits where naive round() would land below it (0.123449 -> 0.1234).
        neg = [i / 10000 for i in range(296)] + [0.123449, 0.5, 0.6, 0.7]
        pos = [0.8 + i * 0.001 for i in range(60)]
        out = recommend_threshold(pos, neg)
        self.assertEqual(out["threshold"], 0.1235)
        self.assertLessEqual(sum(1 for n in neg if n >= out["threshold"]) / len(neg), 0.01)
        self.assertEqual(out["fpr"], 3 / 300)
        self.assertEqual(out["tpr"], 1.0)

    def test_wilson_full_and_middle(self):
        lo, hi = wilson_interval(50, 50)
        self.assertEqual(hi, 1.0)
        self.assertAlmostEqual(lo, 0.9287, places=3)
        lo, hi = wilson_interval(10, 20)
        self.assertAlmostEqual(lo, 0.2993, places=3)
        self.assertAlmostEqual(hi, 0.7007, places=3)

    def test_roc_ties_and_boundary(self):
        neg = [0.1] * 300
        out = recommend_threshold([0.9] * 50, neg)
        self.assertGreater(out["threshold"], 0.1)
        self.assertEqual(out["fpr"], 0.0)

    def test_missing_projects_dir_fails_closed(self):
        from vsk.core import build_project

        with tempfile.TemporaryDirectory() as d:
            gone = str(Path(d, "no_such_dir"))
            with patch.dict(os.environ, {"VSK_PROJECTS_DIR": gone}):
                with self.assertRaises(ConfigError):
                    load_project("ad", root=ROOT)
                with self.assertRaises(ConfigError):
                    build_project("ad", root=ROOT)
                with self.assertRaises(ConfigError):
                    list_projects(root=ROOT)

    def test_cli_flag_beats_env(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            Path(a, "ad.yaml").write_text(PROJECT.format(name="ad"))
            Path(b, "ad.yaml").write_text(
                PROJECT.format(name="ad").replace("face_cosine_min: 0.5", "face_cosine_min: 0.9"))
            with patch.dict(os.environ, {"VSK_PROJECTS_DIR": a}):  # restores env afterwards
                self.assertEqual(load_project("ad", root=ROOT)["qa"]["face_cosine_min"], 0.5)
                buf = io.StringIO()
                with redirect_stdout(buf):
                    self.assertEqual(main(["--root", str(ROOT), "--projects-dir", b, "doctor"]), 0)
                rep = json.loads(buf.getvalue())
                self.assertEqual(rep["projects_dir"]["path"], str(Path(b).absolute()))
                self.assertEqual(load_project("ad", root=ROOT)["qa"]["face_cosine_min"], 0.9)

    def test_cli_local_config_flag_beats_env(self):
        with tempfile.TemporaryDirectory() as d:
            one, two = Path(d, "one.yaml"), Path(d, "two.yaml")
            one.write_text("schema_version: 1\ntools:\n  ffmpeg: /one\n")
            two.write_text("schema_version: 1\ntools:\n  ffmpeg: /two\n")
            with patch.dict(os.environ, {"VSK_LOCAL_CONFIG": str(one)}):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--root", str(ROOT), "--local-config", str(two),
                                           "doctor"]), 0)
                self.assertEqual(resolve_path("tools.ffmpeg", root=ROOT), Path("/two"))

    def test_out_path_tricks(self):
        tricky = [str(ROOT / "tests" / ".." / "tmp_dotdot")]
        if os.name == "nt":
            tricky.append(str(ROOT).swapcase() + "\tmp_case")
        for out in tricky:
            with self.subTest(out=out), redirect_stderr(io.StringIO()):
                rc = main(["--root", str(ROOT), "build", "explainer", "--out", out])
                self.assertEqual(rc, 2)
        self.assertFalse((ROOT / "tmp_dotdot").exists())
        self.assertFalse((ROOT / "tmp_case").exists())

    def test_out_symlink_into_repo(self):
        with tempfile.TemporaryDirectory() as d:
            link = Path(d, "link")
            try:
                os.symlink(ROOT, link, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("cannot create symlinks here")
            with redirect_stderr(io.StringIO()):
                rc = main(["--root", str(ROOT), "build", "explainer", "--out", str(link / "tmp_link")])
            self.assertEqual(rc, 2)
            self.assertFalse((ROOT / "tmp_link").exists())

    def test_oblique_threshold_missing_refuses(self):
        from vsk.qa.identity import ThresholdNotConfigured, backend_from_project, threshold_from_project

        project = {"qa": {"face_cosine_min": 0.6, "face_backend": "yunet_sface"}}
        self.assertEqual(threshold_from_project(project), 0.6)
        with self.assertRaises(ThresholdNotConfigured):
            threshold_from_project(project, oblique=True)
        self.assertEqual(backend_from_project(project), "yunet_sface")
        with self.assertRaises(ValueError):
            backend_from_project({"qa": {"face_backend": "nope"}})

    def test_scores_json_missing_key_is_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d, "bad.json")
            bad.write_text(json.dumps({"positive": [0.5]}))
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["qa", "calibrate-face", "--scores", str(bad)]), 2)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["qa", "calibrate-face", "--scores", str(bad),
                                       "--max-fpr", "2"]), 2)
