"""Document tools: read_docx / read_pptx / read_pdf.

Each tool extracts uploaded document text through the standard-library parser,
records the extraction as evidence and returns a bounded preview. Extraction is
deterministic and never consumes model tokens.
"""
from pathlib import Path

from .. import documents
from . import registry


def _asset(engine, args):
    asset = next((item for item in engine.run["assets"] if item["id"] == args.get("asset_id")), None)
    if asset is None:
        raise ValueError("材料不存在，先 inspect")
    return asset


def _read(engine, args, expected):
    asset = _asset(engine, args)
    suffix = Path(asset["name"]).suffix.lower()
    if suffix != expected:
        raise ValueError(f"{asset['name']} 不是 {expected} 文档")
    if asset.get("kind") != "document":
        raise ValueError(f"{asset['name']} 不是已上传的二进制文档")
    payload = (engine.folder / "inputs" / asset["id"]).read_bytes()
    extracted = documents.extract(asset["name"], payload, max_chars=args.get("max_chars") or 6000)
    entry = {"asset_id": asset["id"], "name": asset["name"], "type": extracted["type"],
             "chars": extracted["chars"], "truncated": extracted["truncated"], "note": extracted["note"]}
    engine.run.setdefault("documents", []).append(entry)
    marker = f"asset:{asset['id']}"
    if not any(source.get("url") == marker for source in engine.run["sources"]):
        engine.run["sources"].append({
            "id": f"d{len(engine.run['documents'])}", "title": asset["name"], "url": marker,
            "abstract": extracted["text"][:1800],
            "evidence_scope": f"{extracted['type'].upper()} 文档正文"})
    engine.run["evidence_collected"] = True
    return {"asset_id": asset["id"], "name": asset["name"], "text": extracted["text"],
            "chars": extracted["chars"], "truncated": extracted["truncated"], "note": extracted["note"]}


def read_docx(engine, args):
    return _read(engine, args, ".docx")


def read_pptx(engine, args):
    return _read(engine, args, ".pptx")


def read_pdf(engine, args):
    return _read(engine, args, ".pdf")


for _name, _handler, _label in (("read_docx", read_docx, "Word"), ("read_pptx", read_pptx, "PowerPoint"),
                                ("read_pdf", read_pdf, "PDF")):
    registry.register_tool(_name, _handler,
                           description=f"提取已上传 {_label} 材料的正文文本；图片、扫描件与复杂排版不解析。",
                           arguments={"asset_id": "材料ID，如 a1", "max_chars": 6000},
                           required=["asset_id"])
