"""侧边栏日志：一条记录能按时间倒序读回。"""

import os
import tempfile
import unittest
from pathlib import Path

from researchloop.activity import recent, record


class ActivityTests(unittest.TestCase):
    def test_task_lines_are_listed_newest_first(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["RESEARCHLOOP_LOG"] = str(Path(directory) / "activity.jsonl")
            try:
                record("brief", "拟定计划 · 超时", model="deepseek-r1:14b", elapsed_ms=180000)
                record("operator", "证据层拦截", run_id="run-1", task="lose_gelu_pool")
                rows = recent()
            finally:
                os.environ.pop("RESEARCHLOOP_LOG", None)
        self.assertEqual([row["text"] for row in rows], ["证据层拦截", "拟定计划 · 超时"])
        self.assertEqual(rows[0]["task"], "lose_gelu_pool")
        self.assertEqual(rows[1]["elapsed_ms"], 180000)


if __name__ == "__main__":
    unittest.main()
