"""Parse a settled JSON answer, without treating reasoning text as an action."""
import json
import re


def parse(content):
    text = str(content or "").lstrip("\ufeff").strip()
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ValueError("模型返回了未结束的推理段，未取得可执行的JSON回答")
        text = text[end + len("</think>"):].strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
        if not match:
            raise ValueError("JSON代码围栏未正确结束，不能作为工具决策")
        text = match.group(1).strip()
    return json.loads(text)
