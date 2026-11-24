"""Loudness helpers: command building and parsing only, ffmpeg is never run."""
import unittest

from vsk.post.master import build_apply_cmd, build_measure_cmd, parse_measurement

STDERR = """ffmpeg version x
[Parsed_loudnorm_0 at 0x1]
{
\t"input_i" : "-23.50",
\t"input_tp" : "-4.10",
\t"input_lra" : "6.20",
\t"input_thresh" : "-34.00",
\t"output_i" : "-16.0",
\t"target_offset" : "0.30"
}
"""


class MasterTests(unittest.TestCase):
    def test_measure_command_targets(self):
        cmd = build_measure_cmd("in.mp4", "ffmpeg")
        self.assertIn("loudnorm=I=-16.0:TP=-1.5:LRA=11.0:print_format=json", cmd)
        self.assertEqual(cmd[-3:], ["-f", "null", "-"])

    def test_parse_and_apply(self):
        measured = parse_measurement(STDERR)
        self.assertEqual(measured["input_i"], "-23.50")
        cmd = build_apply_cmd("in.mp4", "out.mp4", measured, "ffmpeg")
        joined = " ".join(cmd)
        self.assertIn("measured_I=-23.50", joined)
        self.assertIn("offset=0.30", joined)
        self.assertIn("-n", cmd)  # never overwrite

    def test_parse_failure(self):
        with self.assertRaises(ValueError):
            parse_measurement("no json here")
