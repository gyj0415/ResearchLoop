"""Domain-neutral tools. Every experiment retains code, inputs and measurements."""
import csv
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

TOOLS = [
    {"name": "plan", "description": "判断现有数据和工具能否回答问题，说明研究方法和缺失条件；不能执行时明确停止。", "arguments": {"feasible": True, "method": "研究设计", "limitations": "局限", "literature_query": "需要的文献关键词，可空"}},
    {"name": "inspect", "description": "查看可用材料、CSV列/样例和本地实验环境；无材料时可设计可复现数值模拟。", "arguments": {}},
    {"name": "literature", "description": "从Crossref检索真实论文元数据。仅有标题不能证明论文结论；摘要可能缺失。", "arguments": {"query": "英文或中文关键词"}},
    {"name": "read", "description": "读取上传的文本/CSV材料，记录证据来源。", "arguments": {"asset_id": "a1"}},
    {"name": "hypothesis", "description": "实验前注册可证伪假设、机制、控制条件及指标判据。", "arguments": {"statement": "假设", "prediction": "预测", "controls": "控制/比较方法", "metric": "实验输出指标名", "relation": "gt/ge/lt/le", "threshold": 0.0}},
    {"name": "experiment", "description": "为已注册假设运行自主编写的受约束Python。提供np,rng,load_csv(id),load_text(id),emit_result。不得读取任意路径或联网。先实际计算metrics，再emit_result({'metrics':metrics})。", "arguments": {"hypothesis_id": "h1", "code": "实验Python代码"}},
    {"name": "verify", "description": "冻结实验代码，用未向模型暴露的新随机种子重跑，固定代码按已注册判据验证。", "arguments": {"experiment_id": "x1"}},
    {"name": "reflect", "description": "根据真实探索与复核结果，决定修正假设继续研究，或结束并报告负/正结果。", "arguments": {"next_action": "revise/finish", "reason": "基于证据的判断"}},
    {"name": "finish", "description": "以真实证据引用结束研究；没有经过独立重跑的实验只能给出未完成报告。", "arguments": {"conclusion": "结论及局限", "citations": ["h1", "x1", "p1"], "lesson": "有条件的研究方法经验"}},
]


def openalex_abstract(doi, timeout=4):
    """Rebuild an abstract from OpenAlex when Crossref only returned metadata."""
    if not doi:
        return ""
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(str(doi), safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "ResearchLoop/0.4 (local research prototype)"})
    with urllib.request.urlopen(request, timeout=max(1, min(10, timeout))) as response:
        payload = json.load(response)
    inverted = payload.get("abstract_inverted_index") or {}
    positions = {}
    for word, indexes in inverted.items():
        for index in indexes:
            positions[int(index)] = word
    if not positions:
        return ""
    return " ".join(positions[index] for index in sorted(positions))[:1800]


def literature(query, timeout=15):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("需要检索关键词")
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query[:250], "rows": 4})
    request = urllib.request.Request(url, headers={"User-Agent": "ResearchLoop/0.4 (local research prototype)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    rows = []
    for item in payload.get("message", {}).get("items", []):
        abstract = re.sub("<[^>]+>", " ", html.unescape(item.get("abstract", "")))
        rows.append({"title": (item.get("title") or [""])[0], "doi": item.get("DOI"),
                     "url": item.get("URL"), "abstract": abstract[:1800],
                     "evidence_scope": "摘要" if abstract else "只有元数据，不是结论证据"})
    # Crossref often omits abstracts; try OpenAlex for the first few metadata-only records.
    enriched = 0
    for row in rows:
        if enriched >= 4 or row["abstract"]:
            continue
        try:
            abstract = openalex_abstract(row.get("doi"), timeout=4)
        except Exception:
            continue
        if abstract:
            row["abstract"], row["evidence_scope"] = abstract, "摘要（OpenAlex）"
            enriched += 1
    return rows


def inspect_assets(run, folder):
    import numpy
    from researchloop import documents
    rows = []
    for asset in run["assets"]:
        path = folder / "inputs" / asset["id"]
        item = {k: v for k, v in asset.items() if k != "content"}
        if asset.get("kind") == "document":
            try:
                extracted = documents.extract(asset["name"], path.read_bytes(), max_chars=1800)
                item.update(preview=extracted["text"], document_type=extracted["type"],
                            document_note=extracted["note"])
            except Exception as error:
                item.update(preview=f"无法解析：{error}", document_note=str(error))
        elif asset["name"].lower().endswith(".csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                item["columns"], item["preview"] = reader.fieldnames, [row for _, row in zip(range(5), reader)]
        else:
            item["preview"] = path.read_text(encoding="utf-8-sig")[:1800]
        rows.append(item)
    return {"assets": rows, "python": sys.version.split()[0], "numpy": numpy.__version__,
            "capabilities": "文献元数据检索、文本/CSV读取、PDF/DOCX/PPTX文档提取、NumPy数值实验、固定判据的独立随机重跑。没有真实数据时只能研究明确声明的模拟情景。"}


def experiment(code, run, folder, experiment_id, phase, timeout, seed):
    from .worker import validate
    validate(code)
    directory = folder / "experiments" / experiment_id / phase
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"code": code, "seed": seed, "phase": phase,
               "assets": {a["id"]: str(folder / "inputs" / a["id"]) for a in run["assets"]}}
    (directory / "code.py").write_text(code, encoding="utf-8")
    (directory / "input.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    started = time.perf_counter()
    environment = {key: value for key, value in os.environ.items() if key.upper() in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP")}
    environment["PYTHONIOENCODING"] = "utf-8"
    worker = folder / "source" / "researchloop" / "engine" / "worker.py"
    if not worker.exists():
        worker = Path(__file__).with_name("worker.py")
    process = subprocess.run([sys.executable, "-I", "-X", "utf8", "-u", str(worker), str(directory / "input.json")],
                             cwd=directory, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=max(1, min(45, timeout)))
    output = process.stdout[-24000:]
    (directory / "stdout.txt").write_text(output, encoding="utf-8")
    (directory / "stderr.txt").write_text(process.stderr[-12000:], encoding="utf-8")
    if process.returncode or "RESEARCH_RESULT=" not in output:
        raise ValueError((output.rsplit("RESEARCH_ERROR=", 1)[-1] if "RESEARCH_ERROR=" in output else process.stderr or output)[-2000:])
    result = json.loads(output.rsplit("RESEARCH_RESULT=", 1)[-1].strip())
    result.update({"elapsed_seconds": round(time.perf_counter() - started, 3), "phase": phase,
                   "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "seed": seed})
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def check(hypothesis, result):
    value = result["metrics"].get(hypothesis["metric"])
    if value is None:
        raise ValueError(f"实验没有返回预注册指标 {hypothesis['metric']}；实际 metrics：{sorted(result['metrics'])}")
    target = hypothesis["threshold"]
    passed = {"gt": value > target, "ge": value >= target, "lt": value < target, "le": value <= target}[hypothesis["relation"]]
    return {"passed": passed, "metric": hypothesis["metric"], "value": value,
            "relation": hypothesis["relation"], "threshold": target}
