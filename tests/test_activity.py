"""Invocation provenance is appended without losing the underlying measurements."""

import os
import json
import tempfile
import unittest
from pathlib import Path

from researchloop.activity import record
from researchloop import storage


class ActivityTests(unittest.TestCase):
    def test_task_events_are_written_in_actual_order(self):
        with tempfile.TemporaryDirectory(dir=storage.ROOT / '.cache') as directory:
            os.environ["RESEARCHLOOP_LOG"] = str(Path(directory) / "activity.jsonl")
            try:
                record("brief", "拟定计划 · 超时", model="deepseek-r1:14b", elapsed_ms=180000)
                record("operator", "证据层拦截", run_id="run-1", task="lose_gelu_pool")
                rows = [json.loads(line) for line in (Path(directory)/'activity.jsonl').read_text(encoding='utf-8').splitlines()]
            finally:
                os.environ.pop("RESEARCHLOOP_LOG", None)
        self.assertEqual([row["text"] for row in rows], ["拟定计划 · 超时", "证据层拦截"])
        self.assertEqual(rows[1]["task"], "lose_gelu_pool")
        self.assertEqual(rows[0]["elapsed_ms"], 180000)


if __name__ == "__main__":
    unittest.main()
