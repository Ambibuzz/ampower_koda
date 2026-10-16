"""Recording check_page runs: the recorder, the files it writes, and pruning old recordings.

Run from the app folder with the bench's Python: ../../env/bin/python -m unittest discover tests
"""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from ampower_koda.agent.recording import bundle, cleanup, recorder


class FakePage:
    """Answers the recorder's page.evaluate calls the way capture.js does."""

    def __init__(self):
        self.pending = []
        self.truncated = False
        self.stamps = []

    def evaluate(self, script, argument=None):
        if "drain" in script:
            events = self.pending
            self.pending = []
            return {"events": events, "truncated": self.truncated}
        if "stamp" in script:
            self.stamps.append(argument[0])
        return None


class TemporaryFolderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class RecorderTest(TemporaryFolderTest):
    def test_writes_events_steps_and_summary(self):
        page = FakePage()
        recording = recorder.Recorder(self.root)
        recording.page = page
        page.pending = [{"type": 2}]
        recording.drain()
        recording.step_start(1, {"click": "Save"})
        page.pending = [{"type": 3}]
        recording.step_end({"outcome": "OK", "lines": ["saved"]})
        recording.finish("/desk/board")

        record = bundle.load(self.root)
        self.assertEqual(record["events"], [{"type": 2}, {"type": 3}])
        self.assertEqual(record["steps"], [{"index": 1, "action": {"click": "Save"}, "outcome": "OK", "lines": ["saved"]}])
        self.assertEqual(page.stamps, ["koda-step", "koda-step-end"])
        summary = bundle.read_summary(self.root)
        self.assertEqual(summary["route"], "/desk/board")
        self.assertFalse(summary["truncated"])
        self.assertEqual(summary["steps"], [{"index": 1, "outcome": "OK"}])

    def test_keeps_the_start_and_marks_a_long_recording_truncated(self):
        page = FakePage()
        recording = recorder.Recorder(self.root)
        recording.page = page
        with mock.patch.object(recorder, "MAX_EVENTS", 3):
            page.pending = [{"type": 4}, {"type": 2}]
            recording.drain()
            page.pending = [{"type": 3, "n": 1}, {"type": 3, "n": 2}]
            recording.drain()
            page.pending = [{"type": 3, "n": 3}]
            recording.drain()
            recording.finish("/desk/board")

        record = bundle.load(self.root)
        self.assertEqual(record["events"], [{"type": 4}, {"type": 2}, {"type": 3, "n": 1}])
        self.assertTrue(record["meta"]["truncated"])
        self.assertTrue(bundle.read_summary(self.root)["truncated"])

    def test_marks_truncated_when_the_page_stopped_recording(self):
        page = FakePage()
        recording = recorder.Recorder(self.root)
        recording.page = page
        page.pending = [{"type": 2}]
        page.truncated = True
        recording.drain()
        recording.finish("/desk/board")

        self.assertEqual(bundle.load(self.root)["events"], [{"type": 2}])
        self.assertTrue(bundle.read_summary(self.root)["truncated"])

    def test_reports_a_missing_rrweb(self):
        with mock.patch.object(recorder, "node_module_file", return_value=None):
            with self.assertRaises(recorder.RecordingUnavailable) as raised:
                recorder.init_script()
        self.assertIn("bench setup requirements --node", str(raised.exception))


class BundleTest(TemporaryFolderTest):
    def test_round_trip(self):
        record = {"version": 1, "meta": {"route": "/desk/board"}, "events": [{"type": 2}], "steps": []}
        bundle.write(self.root, record)
        bundle.write_summary(self.root, {"route": "/desk/board"})
        self.assertEqual(bundle.load(self.root), record)
        self.assertEqual(bundle.read_summary(self.root), {"route": "/desk/board"})

    def test_missing_files_read_as_empty(self):
        self.assertIsNone(bundle.read_summary(self.root))
        self.assertEqual(bundle.read_error(self.root), "")

    def test_reads_the_error(self):
        (self.root / bundle.ERROR_FILE).write_text("rrweb missing", encoding="utf-8")
        self.assertEqual(bundle.read_error(self.root), "rrweb missing")


class PruneTest(TemporaryFolderTest):
    def make_recording(self, request, name, age_days=0.0, finished=True):
        folder = self.root / request / name
        folder.mkdir(parents=True)
        then = time.time() - age_days * 86400
        if finished:
            (folder / bundle.RECORDING_FILE).write_bytes(b"x")
            os.utime(folder / bundle.RECORDING_FILE, (then, then))
        os.utime(folder, (then, then))
        return folder

    def test_keeps_the_newest_per_request(self):
        for second in range(5):
            self.make_recording("AGENT-0001", f"20261014-12000{second}-ab12-desk_board")
        deleted = cleanup.prune(self.root, keep_per_request=3)
        kept = sorted(path.name for path in (self.root / "AGENT-0001").iterdir())
        self.assertEqual(deleted, 2)
        self.assertEqual(kept, ["20261014-120002-ab12-desk_board", "20261014-120003-ab12-desk_board",
                                "20261014-120004-ab12-desk_board"])

    def test_deletes_old_recordings(self):
        self.make_recording("AGENT-0001", "20260901-120000-ab12-desk_board", age_days=30)
        self.make_recording("AGENT-0002", "20261014-120000-ab12-desk_board")
        self.assertEqual(cleanup.prune(self.root), 1)
        self.assertTrue((self.root / "AGENT-0002").exists())

    def test_leaves_a_running_check_alone_and_deletes_an_abandoned_one(self):
        running = self.make_recording("AGENT-0001", "20261014-120000-ab12-desk_board", finished=False)
        abandoned = self.make_recording("AGENT-0001", "20261013-120000-ab12-desk_board", age_days=1, finished=False)
        cleanup.prune(self.root)
        self.assertTrue(running.exists())
        self.assertFalse(abandoned.exists())

    def test_removes_a_request_folder_left_empty(self):
        self.make_recording("AGENT-0001", "20260901-120000-ab12-desk_board", age_days=30)
        (self.root / "AGENT-0002").mkdir()
        cleanup.prune(self.root)
        self.assertFalse((self.root / "AGENT-0001").exists())
        self.assertFalse((self.root / "AGENT-0002").exists())

    def test_a_missing_root_is_not_an_error(self):
        self.assertEqual(cleanup.prune(self.root / "absent"), 0)


if __name__ == "__main__":
    unittest.main()
