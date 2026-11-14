"""Text extraction for uploaded .docx / .pptx / .pdf materials.

Office formats are ZIP + OOXML, so the standard library is enough. PDF text is
best-effort: only uncompressed or FlateDecode content streams with literal
strings are decoded; scanned pages, encrypted files and exotic font encodings
are reported as limitations instead of being silently fabricated.
"""
import io
import re
import zipfile
import zlib
from xml.etree import ElementTree

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAW = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
CORE = "{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}"

SUFFIXES = {".pdf", ".docx", ".pptx"}
MAX_CHARS = 6000
HARD_MAX_CHARS = 20000


def _truncate(text, max_chars):
    text = text.strip()
    limit = max(200, min(int(max_chars or MAX_CHARS), HARD_MAX_CHARS))
    truncated = len(text) > limit
    return (text[:limit] if truncated else text), truncated


def _clean(text):
    lines = []
    for line in str(text).replace("\r", "\n").split("\n"):
        clean = " ".join(line.split())
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def _xml(data, name):
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise ValueError(f"{name} 结构解析失败：{error}") from error


def docx_text(payload):
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError(f"不是有效的 docx 文件：{error}") from error
    try:
        body = archive.read("word/document.xml")
    except KeyError as error:
        raise ValueError("docx 缺少 word/document.xml") from error
    root = _xml(body, "Word 文档")
    lines = []
    for paragraph in root.iter(f"{WORD}p"):
        pieces = [node.text or "" for node in paragraph.iter(f"{WORD}t")]
        text = "".join(pieces).strip()
        if text:
            lines.append(text)
    title = ""
    try:
        core = _xml(archive.read("docProps/core.xml"), "docProps/core.xml")
        title = (core.findtext(f"{CORE}title") or "").strip()
    except (KeyError, ValueError):
        pass
    return {"type": "docx", "title": title, "text": _clean("\n".join(lines)),
            "note": "仅提取正文段落文本；图片、批注与复杂表格样式不解析。"}


def pptx_text(payload):
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError(f"不是有效的 pptx 文件：{error}") from error
    slides = {}
    for name in archive.namelist():
        match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
        if match:
            slides[int(match.group(1))] = name
    if not slides:
        raise ValueError("pptx 中没有幻灯片")
    blocks = []
    for index in sorted(slides):
        root = _xml(archive.read(slides[index]), f"幻灯片 {index}")
        texts = []
        for shape in root.iter(f"{DRAW}sp"):
            pieces = [node.text or "" for node in shape.iter(f"{DRAW}t")]
            text = "".join(pieces).strip()
            if text:
                texts.append(text)
        if not texts:
            texts = [node.text or "" for node in root.iter(f"{DRAW}t")]
        content = _clean("\n".join(texts))
        if content:
            blocks.append(f"## 幻灯片 {index}\n{content}")
    title = ""
    try:
        core = _xml(archive.read("docProps/core.xml"), "docProps/core.xml")
        title = (core.findtext(f"{CORE}title") or "").strip()
    except (KeyError, ValueError):
        pass
    return {"type": "pptx", "title": title, "text": "\n\n".join(blocks),
            "note": "按幻灯片顺序提取文本框文字；图表、图片与演讲者备注不解析。"}


def _pdf_unescape(value):
    result, index = [], 0
    while index < len(value):
        char = value[index:index + 1]
        if char != b"\\":
            result.append(char)
            index += 1
            continue
        index += 1
        if index >= len(value):
            break
        code = value[index:index + 1]
        index += 1
        mapping = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f",
                   b"(": b"(", b")": b")", b"\\": b"\\"}
        if code in mapping:
            result.append(mapping[code])
        elif code.isdigit():
            digits = code
            for _ in range(2):
                if index < len(value) and value[index:index + 1].isdigit():
                    digits += value[index:index + 1]
                    index += 1
            try:
                result.append(bytes([int(digits, 8) & 0xFF]))
            except ValueError:
                pass
        else:
            result.append(code)
    return b"".join(result)


def _pdf_literal_strings(chunk):
    values = []
    for match in re.finditer(rb"\((?:\\.|[^\\()])*\)", chunk, re.S):
        raw = match.group(0)[1:-1]
        values.append(_pdf_unescape(raw))
    for match in re.finditer(rb"<([0-9A-Fa-f\s]+)>", chunk):
        hexdigits = re.sub(rb"\s+", b"", match.group(1))
        if len(hexdigits) % 2:
            hexdigits += b"0"
        try:
            raw = bytes.fromhex(hexdigits.decode("ascii"))
        except ValueError:
            continue
        try:
            values.append(raw.decode("utf-16-be"))
        except UnicodeDecodeError:
            values.append(raw.decode("latin-1", errors="replace"))
    return values


def _decode_pdf_value(raw):
    if not raw:
        return ""
    for encoding in ("utf-8", "utf-16-be", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if any(ord(ch) < 0x20 and ch not in "\n\t" for ch in text):
            continue
        return text
    return raw.decode("latin-1", errors="replace")


def pdf_text(payload):
    if b"/Encrypt" in payload:
        raise ValueError("PDF 已加密，无法提取文本")
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", payload, re.S)
    if not streams:
        raise ValueError("PDF 没有可读取的内容流")
    parts, decoded_streams = [], 0
    for raw in streams:
        chunk = raw
        try:
            chunk = zlib.decompress(raw)
            decoded_streams += 1
        except zlib.error:
            pass
        if b"BT" not in chunk and b"Tj" not in chunk and b"TJ" not in chunk:
            continue
        values = _pdf_literal_strings(chunk)
        texts = [_decode_pdf_value(value) for value in values]
        joined = " ".join(text.strip() for text in texts if text.strip())
        if joined:
            parts.append(joined)
    text = _clean("\n".join(parts))
    if len(text) < 8:
        raise ValueError("PDF 中未找到可提取的文本层（可能是扫描件或自定义字体编码）")
    return {"type": "pdf", "title": "", "text": text,
            "note": f"尽力提取 {decoded_streams} 个压缩内容流中的文本；扫描件、加密文件与复杂字体编码不受支持。"}


EXTRACTORS = {".docx": docx_text, ".pptx": pptx_text, ".pdf": pdf_text}


def extract(name, payload, max_chars=MAX_CHARS):
    suffix = "." + str(name).rsplit(".", 1)[-1].lower() if "." in str(name) else ""
    if suffix not in EXTRACTORS:
        raise ValueError(f"不支持的文档类型：{suffix or name}")
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("文档内容必须是二进制数据")
    result = EXTRACTORS[suffix](bytes(payload))
    text, truncated = _truncate(result["text"], max_chars)
    return {**result, "text": text, "truncated": truncated, "chars": len(result["text"])}
