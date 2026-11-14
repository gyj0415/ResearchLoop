"""Platform source adapters: public API parsing and site-restricted fallback."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from researchloop import platforms, storage


class BilibiliTests(unittest.TestCase):
    def payload(self):
        return {"code": 0, "data": {"result": [
            {"bvid": "BV1xx", "title": '<em class="keyword">\u5976\u9f99</em>\u8868\u60c5\u5305',
             "description": "video desc", "author": "up", "play": 1234, "video_review": 12,
             "pubdate": 1700000000},
            {"bvid": "", "title": "no id"},
        ]}}

    def test_public_search_is_parsed_and_cleaned(self):
        with patch("researchloop.platforms.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps(self.payload()).encode())):
            result = platforms.search("bilibili", "\u5976\u9f99", limit=5)
        self.assertEqual(result["label"], "\u54d4\u54e9\u54d4\u54e9")
        self.assertEqual(len(result["results"]), 1)
        row = result["results"][0]
        self.assertEqual(row["url"], "https://www.bilibili.com/video/BV1xx")
        self.assertEqual(row["title"], "\u5976\u9f99\u8868\u60c5\u5305")
        self.assertEqual(row["metrics"]["play"], 1234)
        self.assertTrue(row["published_at"])

    def test_api_error_is_reported_not_guessed(self):
        with patch("researchloop.platforms.urllib.request.urlopen",
                   return_value=io.BytesIO(json.dumps({"code": -400, "message": "bad"}).encode())):
            with self.assertRaises(ValueError):
                platforms.search("bilibili", "\u5976\u9f99")


class SiteSearchTests(unittest.TestCase):
    def web(self, enabled=True):
        return {"enabled": enabled, "search_provider": "searxng",
                "search_endpoint": "https://search.example.com" if enabled else "",
                "search_api_key": "", "max_results": 5, "fetch_max_chars": 6000}

    def test_platform_without_public_api_requires_a_search_provider(self):
        with patch("researchloop.storage.settings", return_value={"web": self.web(False)}):
            with self.assertRaises(ValueError) as caught:
                platforms.search("xiaohongshu", "\u5976\u9f99")
        self.assertIn("site:xiaohongshu.com", str(caught.exception))

    def test_site_restricted_search_maps_results(self):
        payload = {"provider": "searxng", "query": "x", "results": [
            {"title": "note", "url": "https://www.xiaohongshu.com/explore/1",
             "snippet": "snippet", "source": "xiaohongshu.com", "published_at": ""}]}
        with patch("researchloop.storage.settings", return_value={"web": self.web(True)}), \
                patch("researchloop.webaccess.search", return_value=payload) as search:
            result = platforms.search("xiaohongshu", "\u5976\u9f99", limit=3)
        self.assertEqual(result["results"][0]["platform"], "xiaohongshu")
        self.assertEqual(result["results"][0]["url"], "https://www.xiaohongshu.com/explore/1")
        self.assertIn("\u7ad9\u5185\u9650\u5b9a", result["note"])
        self.assertIn("site:xiaohongshu.com", search.call_args.args[0])

    def test_tieba_falls_back_to_site_search_when_direct_page_is_blocked(self):
        payload = {"provider": "tavily", "query": "x", "results": [
            {"title": "post", "url": "https://tieba.baidu.com/p/1", "snippet": "", "source": "tieba",
             "published_at": ""}]}
        with patch("researchloop.platforms._tieba", side_effect=ValueError("blocked")), \
                patch("researchloop.storage.settings", return_value={"web": self.web(True)}), \
                patch("researchloop.webaccess.search", return_value=payload):
            result = platforms.search("tieba", "\u5976\u9f99")
        self.assertEqual(result["results"][0]["platform"], "tieba")
        self.assertIn("\u53cd\u722c", result["note"])


class ToolIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_platform_tool_records_sources_with_platform_scope(self):
        from researchloop.engine import core
        engine = core.create({"question": "\u7814\u7a76\u5976\u9f99\u5728\u5e73\u53f0\u4e0a\u7684\u4f20\u64ad"})
        payload = {"platform": "xiaohongshu", "label": "\u5c0f\u7ea2\u4e66", "query": "\u5976\u9f99", "note": "n",
                   "results": [{"platform": "xiaohongshu", "title": "note", "url": "https://www.xiaohongshu.com/explore/1",
                                "snippet": "s", "author": "", "metrics": {}, "published_at": ""}]}
        with patch("researchloop.platforms.search", return_value=payload):
            observation = engine.apply("platform_search",
                                       {"platform": "xiaohongshu", "query": "\u5976\u9f99", "reason": "r"})
        self.assertEqual(observation["platform"], "xiaohongshu")
        source = engine.run["sources"][0]
        self.assertIn("\u5c0f\u7ea2\u4e66", source["evidence_scope"])
        self.assertEqual(source["url"], "https://www.xiaohongshu.com/explore/1")


if __name__ == "__main__":
    unittest.main()

class WebSearchToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_search_deduplicates_urls_and_marks_evidence_scope(self):
        from researchloop.engine import core
        engine = core.create({"question": "研究平台内容如何被检索"})
        engine.run["sources"] = [{"id": "p0", "url": "https://a.example/1", "title": "old"}]
        payload = {"provider": "tavily", "query": "q", "results": [
            {"title": "dup", "url": "https://a.example/1", "snippet": "s"},
            {"title": "new", "url": "https://b.example/2", "snippet": "s"}]}
        web = {"enabled": True, "search_provider": "tavily", "search_endpoint": "https://api.tavily.com/search",
               "search_api_key": "k" * 12, "max_results": 5, "fetch_max_chars": 6000}
        with patch("researchloop.storage.settings", return_value={"web": web}),                 patch("researchloop.webaccess.search", return_value=payload):
            observation = engine.apply("web_search", {"query": "q", "reason": "r"})
        self.assertEqual(observation["added"], 1)
        self.assertEqual(observation["skipped_duplicates"], 1)
        self.assertTrue(engine.run["sources"][1]["evidence_scope"].startswith("网页搜索摘要（tavily）"))
