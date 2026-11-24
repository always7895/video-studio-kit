"""ComfyClient against a fake opener: no sockets."""
import io
import json
import unittest

from vsk.core import ComfyClient, ComfyError, require_idle


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_for(payload):
    def opener(request, timeout=0):
        return FakeResponse(json.dumps(payload).encode())
    return opener


class ComfyClientTests(unittest.TestCase):
    def test_non_loopback_rejected(self):
        for url in ("http://example.com:8188", "https://127.0.0.1:8188", "http://127.0.0.1"):
            with self.assertRaises(ComfyError):
                ComfyClient(url)

    def test_queue_snapshot_and_idle_guard(self):
        idle = {"queue_running": [], "queue_pending": []}
        busy = {"queue_running": [[1]], "queue_pending": []}
        self.assertTrue(require_idle(ComfyClient("http://127.0.0.1:8188", opener_for(idle)).queue_snapshot()).idle)
        with self.assertRaises(ComfyError):
            require_idle(ComfyClient("http://127.0.0.1:8188", opener_for(busy)).queue_snapshot())

    def test_unsafe_output_names_rejected(self):
        client = ComfyClient("http://127.0.0.1:8188", opener_for({}))
        with self.assertRaises(ComfyError):
            client.output_bytes(filename="../x.png")
        with self.assertRaises(ComfyError):
            client.output_bytes(filename="a.png", subfolder="../up")
