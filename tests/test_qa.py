"""QA helpers on synthetic arrays. Needs numpy (qa extra); opencv tests skip without it."""
import unittest

try:
    import numpy as np
except ImportError:  # core-only install
    np = None


class PureNumpyQa(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest("numpy not installed")

    def test_cosine_and_identity_flag(self):
        from vsk.qa.identity import cosine, score_frames, score_identity

        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        self.assertAlmostEqual(cosine(a, a), 1.0)
        self.assertAlmostEqual(cosine(a, b), 0.0)
        self.assertFalse(score_identity([a], a, 0.5)["flag_for_review"])
        out = score_frames([a], {0: a, 5: b}, 0.5)
        self.assertEqual(out["low_frames"], [5])
        self.assertEqual(out["disposition"], "HOLD")
        with self.assertRaises(ValueError):
            cosine(np.zeros(2), a)

    def test_half_rate_detection(self):
        from vsk.qa.dup import alt_ratio, is_half_rate

        smooth = [5.0] * 20
        doubled = [0.0, 10.0] * 10
        self.assertGreater(alt_ratio(smooth), 0.9)
        self.assertFalse(is_half_rate(smooth))
        self.assertTrue(is_half_rate(doubled))

    def test_cuts_freezes_and_jumps(self):
        from vsk.qa.cuts import detect_cuts, freeze_runs, frame_diffs, in_shot_jumps

        diffs = [1.0] * 10 + [40.0] + [1.0] * 10
        self.assertEqual(detect_cuts(diffs), [10])
        self.assertEqual(in_shot_jumps(diffs, cuts=[10]), [])
        self.assertEqual(in_shot_jumps(diffs, cuts=[]), [10])
        self.assertEqual(freeze_runs([5, 0.0, 0.0, 0.0, 5, 0.1, 5]), [(1, 4)])
        frames = [np.zeros((4, 4)), np.full((4, 4), 8.0)]
        self.assertEqual(frame_diffs(frames), [8.0])

    def test_background_mask(self):
        from vsk.qa.flow import background_mask

        mask = background_mask((10, 10), [(2, 2, 6, 6)], margin=0.0)
        self.assertFalse(mask[3, 3])
        self.assertTrue(mask[0, 0])


class OpenCvQa(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest("numpy not installed")

    def test_background_flow_detects_global_shift(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv not installed")
        rng = np.random.default_rng(0)
        base = cv2.GaussianBlur((rng.random((96, 96)) * 255).astype(np.uint8), (0, 0), 3)
        shifted = np.roll(base, 3, axis=1)
        from vsk.qa.flow import background_flow

        self.assertLess(background_flow(base, base), 0.05)
        self.assertGreater(background_flow(base, shifted), 0.5)


class ThresholdFailClosed(unittest.TestCase):
    def setUp(self):
        if np is None:
            self.skipTest("numpy not installed")

    def test_missing_threshold_refuses(self):
        from vsk.qa.identity import (ThresholdNotConfigured, score_frames,
                                     score_identity, threshold_from_project)

        a = np.array([1.0, 0.0])
        with self.assertRaises(ThresholdNotConfigured):
            score_identity([a], a)
        with self.assertRaises(ThresholdNotConfigured):
            score_frames([a], {0: a}, None)
        with self.assertRaises(ThresholdNotConfigured):
            threshold_from_project({"qa": {"face_cosine_min": None}})
        with self.assertRaises(ThresholdNotConfigured):
            score_identity([a], a, True)
        self.assertEqual(threshold_from_project({"qa": {"face_cosine_min_oblique": 0.5}},
                                                oblique=True), 0.5)

    def test_backend_registry(self):
        from vsk.qa.identity import BACKENDS, make_backend

        self.assertEqual(set(BACKENDS), {"insightface", "yunet_sface"})
        with self.assertRaises(ValueError):
            make_backend("nope")
