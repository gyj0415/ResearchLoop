import json
import time
import urllib.error
import urllib.parse
import urllib.request


class BudgetError(RuntimeError):
    pass


class Budget:
    def __init__(self, token_limit, seconds_limit):
        self.token_limit = token_limit
        self.seconds_limit = seconds_limit
        self.started = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.unknown_reserved = 0
        self.calls = []

    @property
    def used(self):
        return self.input_tokens + self.output_tokens + self.unknown_reserved

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    def reserve(self, amount):
        if self.used + amount > self.token_limit:
            raise BudgetError("剩余 token 不足以保守预留下一次调用")
        if self.elapsed >= self.seconds_limit - 2:
            raise BudgetError("研究时间预算已耗尽")

    def record(self, item):
        self.calls.append(item)
        self.input_tokens += item.get("input_tokens", 0)
        self.output_tokens += item.get("output_tokens", 0)
        self.unknown_reserved += item.get("unknown_reserved", 0)

    def snapshot(self):
        return {"token_limit": self.token_limit, "seconds_limit": self.seconds_limit,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "unknown_reserved": self.unknown_reserved, "total_tokens": self.used,
                "elapsed_seconds": round(self.elapsed, 3), "model_calls": len(self.calls),
                "cloud_calls": 0, "cloud_cost": 0,
                "note": "Logical prompt + generated tokens; unknown usage consumes reserved allowance. Tool time is recorded separately."}


def probe(base_url):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=3) as response:
            result = json.load(response)
        models = [{"name": m["name"], "size": m.get("size"), "digest": m.get("digest"),
                   "details": m.get("details", {})} for m in result.get("models", [])]
        return {"available": True, "models": models, "error": None}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}


class Ollama:
    def __init__(self, base_url, model, budget, on_call):
        if "cloud" in model.lower():
            raise ValueError("当前版本只允许本地模型，不能使用 cloud 模型")
        self.base_url, self.model, self.budget, self.on_call = base_url.rstrip("/"), model, budget, on_call
        self.model_digest = None

    def call(self, prompt, schema, purpose, seed):
        context, output = 4096, 768
        reserve = context + output
        self.budget.reserve(reserve)
        body = {"model": self.model, "stream": False, "think": False, "format": schema,
                "messages": [{"role": "system", "content": "You research RETRIEVAL ALGORITHMS, not biology. Scientific claims are evaluation queries, NOT scientific hypotheses for you to prove. Choose one allowed retrieval experiment or stop. Return only the requested JSON, with short Chinese explanations. Never invent measurements."},
                             {"role": "user", "content": prompt}],
                "options": {"temperature": 0, "seed": seed, "num_ctx": context, "num_predict": output},
                "keep_alive": "10m"}
        started = time.monotonic()
        event = {"purpose": purpose, "model": self.model, "reserved_upper_bound": reserve,
                 "prompt": prompt, "request_options": body["options"], "cache_hit": False}
        try:
            request = urllib.request.Request(self.base_url + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            timeout = max(1, min(180, self.budget.seconds_limit - self.budget.elapsed))
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            if "prompt_eval_count" not in result or "eval_count" not in result:
                raise ValueError("Ollama 未返回完整 token usage，已按预留量记账并停止")
            event.update({"input_tokens": result["prompt_eval_count"], "output_tokens": result["eval_count"],
                          "cached_prompt_tokens": result.get("prompt_eval_cached_count", 0),
                          "server_seconds": result.get("total_duration", 0) / 1e9,
                          "load_seconds": result.get("load_duration", 0) / 1e9,
                          "raw_response": result, "status": "ok"})
        except Exception as exc:
            event.update({"status": "error", "error": str(exc), "unknown_reserved": reserve})
            raise RuntimeError(f"Ollama 调用失败：{exc}") from exc
        finally:
            event["wall_seconds"] = time.monotonic() - started
            self.budget.record(event)
            self.on_call(event)
        if self.budget.used > self.budget.token_limit:
            raise BudgetError("模型实际 usage 超过预留上界，停止后续调用")
        content = result.get("message", {}).get("content", "").strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("模型未返回有效 JSON；该调用的全部消耗已计入") from exc
