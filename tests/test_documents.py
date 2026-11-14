"""Document extraction contracts: docx / pptx / pdf tools and upload validation."""
import base64
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from researchloop import documents, storage
from researchloop.engine import core


def make_docx(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<?xml version='1.0'?><Types/>")
        archive.writestr("word/document.xml",
                         "<?xml version='1.0'?><w:document "
                         "xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
                         f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>")
        archive.writestr("docProps/core.xml",
                         "<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'>"
                         "<cp:title>测试文档</cp:title></cp:coreProperties>")
    return buffer.getvalue()


def make_pptx(text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<?xml version='1.0'?><Types/>")
        archive.writestr("ppt/slides/slide1.xml",
                         "<?xml version='1.0'?><p:sld xmlns:p='http://schemas.openxmlformats.org/presentationml/2006/main' "
                         "xmlns:a='http://schemas.openxmlformats.org/drawingml/2006/main'>"
                         f"<p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:sld>")
    return buffer.getvalue()


def make_pdf(text):
    return (b"%PDF-1.4\n1 0 obj << /Length 60 >>\nstream\n"
            b"BT /F1 12 Tf 72 720 Td (" + text.encode() + b") Tj ET\n"
            b"endstream\nendobj\ntrailer << /Root 1 0 R >>\n%%EOF\n")


class ExtractionTests(unittest.TestCase):
    def test_docx_text_and_title(self):
        result = documents.extract("report.docx", make_docx("样本量越大误差越小"))
        self.assertEqual(result["type"], "docx")
        self.assertIn("样本量越大误差越小", result["text"])
        self.assertEqual(result["title"], "测试文档")

    def test_pptx_slide_marker(self):
        result = documents.extract("deck.pptx", make_pptx("方法：重复模拟"))
        self.assertIn("幻灯片 1", result["text"])
        self.assertIn("重复模拟", result["text"])

    def test_pdf_literal_text(self):
        result = documents.extract("paper.pdf", make_pdf("Hello PDF text"))
        self.assertIn("Hello PDF text", result["text"])
        self.assertIn("尽力提取", result["note"])

    def test_encrypted_pdf_is_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            documents.extract("secret.pdf", b"%PDF-1.4 /Encrypt 5 0 R\nstream\nx\nendstream")

    def test_truncation_and_unknown_suffix(self):
        result = documents.extract("long.docx", make_docx("填充" * 4000), max_chars=300)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["text"]), 300)
        with self.assertRaises(ValueError):
            documents.extract("file.xlsx", b"x")


class UploadAndToolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=storage.ROOT / ".cache")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name, value in (("RUNS", root / "runs"), ("MEMORY", root / "memory")):
            patcher = patch.object(storage, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def create_with_docx(self):
        encoded = base64.b64encode(make_docx("实验中样本量影响误差")).decode()
        return core.create({"question": "研究样本量对估计误差的影响", "max_steps": 8,
                            "assets": [{"name": "materials.docx", "content_base64": encoded}]})

    def test_document_asset_is_stored_and_extracted_by_the_matching_tool(self):
        engine = self.create_with_docx()
        asset = engine.run["assets"][0]
        self.assertEqual(asset["kind"], "document")
        self.assertTrue((engine.folder / "inputs" / asset["id"]).exists())
        engine.run["evidence_collected"] = True
        tool, automatic = engine.next_protocol_action()
        self.assertEqual(tool, "read_docx")
        self.assertEqual(automatic, {"asset_id": asset["id"]})
        observation = engine.apply(tool, automatic)
        self.assertIn("样本量影响误差", observation["text"])
        self.assertEqual(engine.run["documents"][0]["type"], "docx")
        self.assertTrue(any(source.get("url") == f"asset:{asset['id']}" for source in engine.run["sources"]))

    def test_mismatched_tool_is_rejected(self):
        engine = self.create_with_docx()
        engine.run["evidence_collected"] = True
        with self.assertRaises(ValueError):
            engine.apply("read_pdf", {"asset_id": "a1"})

    def test_binary_upload_requires_a_supported_document_suffix(self):
        encoded = base64.b64encode(b"whatever").decode()
        with self.assertRaises(ValueError):
            core.create({"question": "研究样本量对估计误差的影响", "max_steps": 8,
                         "assets": [{"name": "notes.exe", "content_base64": encoded}]})


class RegistryDocumentTools(unittest.TestCase):
    def test_document_tools_are_registered(self):
        from researchloop.engine import registry
        names = set(registry.names())
        self.assertTrue({"read_docx", "read_pptx", "read_pdf"} <= names)


if __name__ == "__main__":
    unittest.main()
