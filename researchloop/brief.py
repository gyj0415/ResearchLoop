"""把一句研究方向收成可检验的计划。

对话只负责拟定问题、成功标准和否证条件。测量数字只能来自后续实验，
计划文本里不写已经测到的加速比。本地模型不可用时按同一套规则草拟，并标明来源。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

TRANSCENDENTAL = ("gelu", "silu", "softmax", "tanh", "sigmoid", "erf", "exp")
RETRIEVAL = ("检索", "scifact", "bm25", "证据句", "召回")
REDUCE = ("mean", "均值", "偏置", "bias", "归约", "reduce")
FORBIDDEN = ("跳过门控", "不测正确性", "伪造", "直接写成", "不要验证")


def draft(messages, *, task=None, model=None, skip_reason=None):
    """返回一份计划。`model` 为 `{ollama_url, model}` 时先问本地模型，失败则用规则。"""
    started = time.perf_counter()
    base = {"at": datetime.now(timezone.utc).isoformat(),
            "model": (model or {}).get("model") or "",
            "url": (model or {}).get("ollama_url") or ""}
    rules = rules_draft(messages, task)

    def finish(plan, outcome, reason_code, reason_label, detail="", raw=""):
        plan["log"] = {**base, "outcome": outcome, "reason_code": reason_code,
                       "reason_label": reason_label, "detail": str(detail)[:800],
                       "raw_excerpt": str(raw)[:800],
                       "elapsed_ms": int((time.perf_counter() - started) * 1000)}
        return plan

    if not model:
        rules["source"] = "rules"
        label = skip_reason or "未调用模型"
        return finish(rules, "rules", "skipped", label, label)
    raw = ""
    try:
        parsed, raw = _ask_model(messages, task, model)
        plan = normalize(parsed)
    except Exception as exc:
        code, label = _failure(exc)
        rules["source"] = "rules"
        rules["model_error"] = str(exc)
        return finish(rules, "rules", code, label, str(exc), raw)
    plan["source"] = "ollama"
    if task and task.get("name"):
        plan["task_name"] = task["name"]
    return finish(plan, "ollama", "ok", "本地模型拟定", "", raw)


def rules_draft(messages, task=None):
    text = _latest(messages).lower()
    attached = (task or {}).get("name") or ""
    ir = (task or {}).get("ir") or ""
    if any(word in text for word in RETRIEVAL) and not any(word in text for word in TRANSCENDENTAL + REDUCE):
        return _plan(
            "retrieval", "证据检索策略",
            "在已有 SciFact 划分上，哪一种受约束的检索改动能在独立验收里不退化？",
            "验收集的完整命中和覆盖率都不低于固定 BM25。计划不包含任何已测分数。",
            ["用失败类型描述探索集，不把金标原句送进规划", "只从合法单因素配置里选一个实验", "开发筛选之后做一次独立验收", "不达标就不发布经验"],
            ["新语料", "改评估代码", "逐题重排"],
            "独立验收的完整命中或覆盖率低于基线，这个方向就停。",
            "C",
            "这是检索方向。任务用现有的 C，模型只能在封闭配置里选题，不能新造数据集。")
    transcendental = any(word in text or word in ir.lower() for word in TRANSCENDENTAL)
    if transcendental:
        return _plan(
            "operator", "先判断这条算子值不值得做",
            "这个参考实现里的超越函数，会不会让 numpy 候选在实验前就没有胜算？",
            "若微基准显示超越函数是硬上限，则实验数为 0，并给出原因码。不预设加速比。",
            ["读 reference 的结构", "做单算子微基准", "若判定赢不了，就在实验前停止，不生成候选", "只有结构改写说得通时才进入门控和同窗计时"],
            ["手写绕过 torch 的实现", "跳过正确性直接报加速比", "把计划里的判断当成已经测完"],
            "若微基准里的超越函数并没有数量级落后，这条提前停止就不成立，应改做实验。",
            attached,
            "方向先收成「值不值得做」。超越函数默认预期会在实验前停住；这只是计划，不是测量结果。")
    if any(word in text or word in ir.lower() for word in REDUCE):
        return _plan(
            "operator", "寻找可消元的写法",
            "这个参考实现有没有一种结构不同、但数值仍正确的写法，能在同窗里快于 torch eager？",
            "候选通过 G0–G4，且同窗加速比达到 1.10×。达不到就保持基线。",
            ["从 reference 抬成结构", "先看有没有可消元的归约", "生成候选后先过正确性", "只对通过的候选做同窗计时", "用原因码写下采纳或放弃"],
            ["改计时器", "用未验证的代码报速度", "把例子任务的数字抄到新任务上"],
            "同窗加速比低于 1.10×，或任一正确性门控失败，就不能说这个方向赢了。",
            attached,
            "方向收成一次算子研究。成功与否只看这次的门控和同窗计时，计划里不写已经得到的倍数。")
    return _plan(
        "unscoped", "这句话还没有落到现有实验上",
        text[:180] or "需要把方向收成一个可运行的问题。",
        "先确认它属于算子加速，还是 SciFact 检索。确认前不开跑。",
        ["改写成算子参考实现，或明确说要做证据检索", "再拟定一版计划"],
        ["把未测量的题目写成算子加速结论"],
        "如果它既不是可融合算子，也不是现有检索任务，这个计划不能启动实验。",
        attached,
        "本地模型没有给出合格计划时，这句话不会被当成算子研究。请看标签上的日志。")


def normalize(value):
    if not isinstance(value, dict):
        raise ValueError("计划必须是 JSON 对象")
    domain = value.get("domain")
    if domain not in ("operator", "retrieval", "unscoped"):
        domain = "unscoped"
    steps = _texts(value.get("steps"), 6)
    scope = _texts(value.get("out_of_scope"), 6)
    blob = " ".join(steps + scope + [str(value.get("reply") or ""), str(value.get("success") or "")])
    if any(word in blob for word in FORBIDDEN):
        raise ValueError("计划不能跳过门控或伪造测量")
    if any(phrase in blob for phrase in ("已经达到", "实测为", "已测得")):
        raise ValueError("计划不能提前写入测量结果")
    question = _clip(value.get("question"), 300)
    success = _clip(value.get("success"), 300)
    falsify = _clip(value.get("falsify_if"), 300)
    if not question or not success or not falsify or not steps:
        raise ValueError("计划缺少问题、成功标准、步骤或否证条件")
    task_name = value.get("task_name") or ""
    if not isinstance(task_name, str):
        task_name = ""
    return _plan(domain, _clip(value.get("title"), 80) or "未命名方向", question, success,
                 steps, scope, falsify, task_name[:80], _clip(value.get("reply"), 400) or question)


def _plan(domain, title, question, success, steps, scope, falsify, task_name, reply):
    return {"domain": domain, "title": title, "question": question, "success": success,
            "steps": steps, "out_of_scope": scope, "falsify_if": falsify,
            "task_name": task_name, "reply": reply}


def _ask_model(messages, task, model):
    name = str(model.get("model") or "")
    if not name or "cloud" in name.lower():
        raise ValueError("请选择本地 Ollama 模型")
    url = str(model.get("ollama_url") or "http://127.0.0.1:11434").rstrip("/")
    user = _latest(messages)
    attached = ""
    if task:
        attached = f"\n已加入的任务：{task.get('name') or ''}。结构：{task.get('ir') or '尚未读懂'}。"
    history = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in messages[-6:] if isinstance(item, dict))
    prompt = (
        "根据对话拟定研究计划，不要执行，不要编造测量数字。"
        "算子计划的成功标准只能是：通过正确性门控后的同窗加速比达到 1.10 倍，否则保持基线。"
        "含 GELU、exp、tanh、softmax 时，计划应把「实验前判断赢不了」写进步骤。"
        "检索计划只能使用已有任务 C 或 D，不能新建语料。"
        f"{attached}\n对话：\n{history}\n最新要求：{user}"
    )
    schema = {"type": "object", "properties": {
        "domain": {"type": "string"}, "title": {"type": "string"}, "question": {"type": "string"},
        "success": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}},
        "out_of_scope": {"type": "array", "items": {"type": "string"}},
        "falsify_if": {"type": "string"}, "task_name": {"type": "string"}, "reply": {"type": "string"}},
        "required": ["domain", "title", "question", "success", "steps", "out_of_scope", "falsify_if", "reply"]}
    body = {"model": name, "stream": False, "think": False, "format": schema,
            "messages": [{"role": "system", "content": "你只拟定研究计划。返回 JSON。不要声称已经测到加速比或命中率。"},
                         {"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": 900}}
    request = urllib.request.Request(url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
    except TimeoutError as exc:
        raise TimeoutError("等待本地模型超过 180 秒") from exc
    except urllib.error.URLError as exc:
        if "timed out" in str(exc).lower():
            raise TimeoutError("等待本地模型超过 180 秒") from exc
        raise
    content = result.get("message", {}).get("content", "").strip()
    return _load_json(content), content


def _load_json(content):
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _failure(exc):
    text = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timed out" in text or "超时" in str(exc):
        return "timeout", "超时"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_json", "返回无效"
    if isinstance(exc, ValueError):
        return "rejected", "计划被拒"
    if "refused" in text or "拒绝" in str(exc) or "not found" in text:
        return "unreachable", "未连接"
    return "error", "调用失败"


def _latest(messages):
    for item in reversed(messages or []):
        if isinstance(item, dict) and item.get("role") == "user" and str(item.get("content") or "").strip():
            return str(item["content"]).strip()[:2000]
    raise ValueError("先写一句研究方向")


def _texts(value, limit):
    if not isinstance(value, list):
        return []
    return [_clip(item, 160) for item in value[:limit] if _clip(item, 160)]


def _clip(value, limit):
    text = " ".join(str(value or "").split())
    return text[:limit]
