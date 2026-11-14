"""Archive hides sessions from the sidebar; delete moves them to a recoverable trash."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import storage
from researchloop.engine import core


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.engine = core.create({"question": "研究样本量对估计误差的影响"})
        self.run_id = self.engine.run["id"]

    def test_archive_hides_then_restores(self):
        storage.archive_run(self.run_id)
        runs = {run["id"]: run for run in storage.list_runs()}
        self.assertTrue(runs[self.run_id]["archived"])
        self.assertTrue(runs[self.run_id]["archived_at"])
        storage.archive_run(self.run_id, False)
        runs = {run["id"]: run for run in storage.list_runs()}
        self.assertFalse(runs[self.run_id]["archived"])

    def test_delete_requires_archive_first(self):
        with self.assertRaises(ValueError):
            storage.delete_run(self.run_id)
        self.assertTrue((storage.RUNS / self.run_id).exists())

    def test_delete_moves_the_folder_to_trash_and_hides_it(self):
        storage.archive_run(self.run_id)
        storage.delete_run(self.run_id)
        self.assertFalse((storage.RUNS / self.run_id).exists())
        self.assertTrue((storage.RUNS / ".trash" / self.run_id / "run.json").exists())
        self.assertNotIn(self.run_id, [run["id"] for run in storage.list_runs()])
        with self.assertRaises(ValueError):
            storage.run_path(self.run_id)


if __name__ == "__main__":
    unittest.main()
