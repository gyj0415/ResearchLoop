"""Crossref metadata can be enriched with OpenAlex abstracts instead of pretending none exist."""
import io
import json
import unittest
from unittest.mock import patch

from researchloop.engine import tools


def crossref_payload():
    return {"message": {"items": [
        {"title": ["Metadata only paper"], "DOI": "10.1000/abc", "URL": "https://doi.org/10.1000/abc"},
    ]}}


def openalex_payload():
    return {"abstract_inverted_index": {"Hello": [0], "world": [1]}}


class AbstractEnrichmentTests(unittest.TestCase):
    def test_missing_crossref_abstract_is_filled_from_openalex(self):
        responses = [io.BytesIO(json.dumps(crossref_payload()).encode()),
                     io.BytesIO(json.dumps(openalex_payload()).encode())]
        with patch("researchloop.engine.tools.urllib.request.urlopen", side_effect=responses):
            rows = tools.literature("query")
        self.assertEqual(rows[0]["abstract"], "Hello world")
        self.assertEqual(rows[0]["evidence_scope"], "摘要（OpenAlex）")

    def test_openalex_failure_keeps_metadata_only_scope(self):
        responses = [io.BytesIO(json.dumps(crossref_payload()).encode())]
        def fake_urlopen(request, timeout=None):
            url = request.full_url if hasattr(request, "full_url") else str(request)
            if "openalex" in url:
                raise OSError("network down")
            return responses.pop(0)
        with patch("researchloop.engine.tools.urllib.request.urlopen", side_effect=fake_urlopen):
            rows = tools.literature("query")
        self.assertEqual(rows[0]["abstract"], "")
        self.assertEqual(rows[0]["evidence_scope"], "只有元数据，不是结论证据")


if __name__ == "__main__":
    unittest.main()
