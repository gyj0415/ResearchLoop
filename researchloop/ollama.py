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
        self._frozen_elapsed = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.unknown_reserved = 0
        self.cached_tokens = 0
        self.cloud_calls = 0
        self.cloud_cost_usd = 0.0
        self.cloud_cost_known = True
        self.calls = []

    @property
    def used(self):
        return self.input_tokens + self.output_tokens + self.unknown_reserved

    def restore(self, snapshot, calls):
        """Resume active research time and spending, excluding time spent paused."""
        for key in ("input_tokens", "output_tokens", "unknown_reserved", "cached_tokens", "cloud_calls"):
            setattr(self, key, snapshot.get(key, 0))
        self.cloud_cost_usd = snapshot.get("cloud_cost_usd", snapshot.get("cloud_cost", 0))
        self.cloud_cost_known = snapshot.get("cloud_cost_known", True)
        self.calls = list(calls)
        self.started = time.monotonic() - snapshot.get("elapsed_seconds", 0)

    @property
    def elapsed(self):
        return self._frozen_elapsed if self._frozen_elapsed is not None else time.monotonic() - self.started

    def suspend(self):
        self._frozen_elapsed = self.elapsed

    def resume_clock(self):
        elapsed = self.elapsed
        self.started = time.monotonic() - elapsed
        self._frozen_elapsed = None

    @property
    def token_cap(self):
        return self.token_limit  # None means unlimited

    def token_room(self, amount):
        return self.token_limit is None or self.used + amount <= self.token_limit

    def seconds_left(self):
        return None if self.seconds_limit is None else max(0.0, self.seconds_limit - self.elapsed)

    def reserve(self, amount):
        if not self.token_room(amount):
            raise BudgetError("剩余 token 不足以保守预留下一次调用")
        if self.seconds_limit is not None and self.elapsed >= self.seconds_limit - 2:
            raise BudgetError("研究时间预算已耗尽")

    def record(self, item):
        self.calls.append(item)
        self.input_tokens += item.get("input_tokens", 0)
        self.output_tokens += item.get("output_tokens", 0)
        self.unknown_reserved += item.get("unknown_reserved", 0)
        if item.get("cached_tokens"):
            self.cached_tokens += int(item["cached_tokens"])
        if item.get("wall_seconds") and item.get("output_tokens") and not item.get("tok_per_second"):
            item["tok_per_second"] = round(item["output_tokens"] / max(float(item["wall_seconds"]), 0.001), 3)
        if item.get("cloud"):
            self.cloud_calls += 1
            cost = item.get("cloud_cost")
            if cost is None:
                self.cloud_cost_known = False
            else:
                self.cloud_cost_usd += float(cost)

    def snapshot(self):
        note = "Logical prompt + generated tokens; unknown usage consumes reserved allowance. Tool time is recorded separately."
        if self.cloud_calls:
            note += " 云端调用已计入 token 预算；未配置单价时 cloud_cost 不代表零费用。"
        return {"token_limit": self.token_limit, "seconds_limit": self.seconds_limit,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "unknown_reserved": self.unknown_reserved, "total_tokens": self.used,
                "cached_tokens": self.cached_tokens,
                "tok_per_second": round(self.output_tokens / self.elapsed, 3) if self.elapsed else None,
                "elapsed_seconds": round(self.elapsed, 3), "model_calls": len(self.calls),
                "cloud_calls": self.cloud_calls, "cloud_cost": round(self.cloud_cost_usd, 9),
                "cloud_cost_usd": round(self.cloud_cost_usd, 9),
                "cloud_cost_known": self.cloud_cost_known, "note": note}


def probe(base_url):
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=3) as response:
            result = json.load(response)
        models = [{"name": m["name"], "size": m.get("size"), "digest": m.get("digest"),
                   "details": m.get("details", {})} for m in result.get("models", [])]
        return {"available": True, "models": models, "error": None}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}
