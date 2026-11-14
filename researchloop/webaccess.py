"""Web access for research tools, ported from the ReEscape Protocol design.

`validate_outbound` blocks loopback/private/link-local targets before any
request (redirect targets are re-validated), fetch limits bytes and characters,
and search supports a SearxNG or Tavily endpoint from local settings.
"""
import html
import ipaddress
import json
import socket
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser

USER_AGENT = "ResearchLoopBot/0.1"
MAX_BYTES = 1 << 20
MAX_CHARS = 6000
HARD_MAX_CHARS = 20000
MAX_RESULTS = 10
TIMEOUT = 8

TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json",
              "application/xml", "application/rss+xml", "application/atom+xml")


def _reject(ip, label):
    if (ip.is_loopback or ip.is_private or ip.is_unspecified or ip.is_link_local
            or ip.is_multicast or ip.is_reserved):
        raise ValueError(f"不允许访问{label}地址：{ip}")


def validate_outbound(raw, allowed_hosts=()):
    parsed = urllib.parse.urlparse(str(raw or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("仅允许 http/https URL")
    host = parsed.hostname
    if not host:
        raise ValueError("URL 缺少主机名")
    if host.lower().rstrip(".") == "localhost":
        raise ValueError("不允许访问 localhost")
    if parsed.port not in (None, 80, 443):
        raise ValueError("只允许标准端口 80/443")
    if any(host.lower() == str(item).lower() for item in allowed_hosts):
        return parsed
    try:
        _reject(ipaddress.ip_address(host), "私有或本地")
        return parsed
    except ValueError as error:
        if "不允许访问" in str(error):
            raise
    addresses = socket.getaddrinfo(host, None)
    if not addresses:
        raise ValueError("无法解析主机")
    for info in addresses:
        _reject(ipaddress.ip_address(info[4][0]), "私有或本地")
    return parsed


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.title, self._skip, self._in_title = [], "", 0, False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "section", "article"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip:
            return
        text = html.unescape(data)
        if self._in_title and not self.title:
            self.title = " ".join(text.split())
        self.parts.append(text)


def _normalise(text):
    lines = []
    for line in str(text).replace("\r", "\n").split("\n"):
        clean = " ".join(line.split())
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def fetch(raw, max_chars=MAX_CHARS, timeout=TIMEOUT, user_agent=USER_AGENT, allowed_hosts=()):
    parsed = validate_outbound(raw, allowed_hosts)
    max_chars = max(200, min(int(max_chars or MAX_CHARS), HARD_MAX_CHARS))
    request = urllib.request.Request(parsed.geturl(), headers={
        "Accept": "text/html,text/plain,application/json,application/xml;q=0.8,*/*;q=0.1",
        "User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=max(1, min(30, timeout))) as response:
        validate_outbound(response.geturl(), allowed_hosts)
        media = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read(MAX_BYTES + 1)
    if media not in TEXT_TYPES:
        raise ValueError(f"不支持的响应类型：{media or '未知'}")
    truncated = len(body) > MAX_BYTES
    body = body[:MAX_BYTES]
    text = body.decode(charset, errors="replace")
    if media in ("text/html", "application/xhtml+xml"):
        extractor = _TextExtractor()
        extractor.feed(text)
        title, content = extractor.title, _normalise("".join(extractor.parts))
    else:
        title, content = "", _normalise(text)
    if len(content) > max_chars:
        content, truncated = content[:max_chars], True
    return {"url": response.geturl(), "title": title or urllib.parse.urlparse(response.geturl()).hostname or "",
            "content": content, "content_type": media,
            "fetched_at": datetime.now(timezone.utc).isoformat(), "truncated": truncated}


def _json(url, headers, timeout):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=max(1, min(30, timeout))) as response:
        media = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        body = response.read((2 << 20) + 1)
    if media not in ("application/json", "") and not body.strip().startswith((b"{", b"[")):
        raise ValueError(f"搜索接口未返回 JSON（{media}）")
    return json.loads(body.decode("utf-8", errors="replace"))


def _time_range(recency):
    value = str(recency or "any").strip().lower()
    if value not in ("any", "day", "week", "month", "year"):
        raise ValueError("recency 只能是 any/day/week/month/year")
    return "" if value == "any" else value


def search(query, provider="searxng", endpoint="", api_key="", max_results=5, recency="any", timeout=TIMEOUT):
    text = str(query or "").strip()
    if not text:
        raise ValueError("搜索需要关键词")
    if len(text) > 300:
        raise ValueError("搜索关键词最多 300 字符")
    limit = max(1, min(int(max_results or 5), MAX_RESULTS))
    window = _time_range(recency)
    provider = str(provider or "searxng").strip().lower()
    if provider == "searxng":
        base = str(endpoint or "").strip()
        if not base:
            raise ValueError("未配置 SearxNG 搜索地址")
        target = urllib.parse.urlparse(base)
        values = urllib.parse.parse_qs(target.query)
        values["q"], values["format"] = [text], ["json"]
        if window:
            values["time_range"] = [window]
        url = urllib.parse.urlunparse(target._replace(query=urllib.parse.urlencode(values, doseq=True)))
        payload = _json(url, {"Accept": "application/json", "User-Agent": USER_AGENT}, timeout)
        rows = [{"title": str(item.get("title") or "").strip(), "url": str(item.get("url") or "").strip(),
                 "snippet": str(item.get("content") or "").strip(),
                 "source": str(item.get("engine") or urllib.parse.urlparse(str(item.get("url") or "")).hostname or ""),
                 "published_at": str(item.get("publishedDate") or "").strip()}
                for item in payload.get("results", [])]
    elif provider == "tavily":
        base = str(endpoint or "https://api.tavily.com/search").strip()
        if not str(api_key or "").strip():
            raise ValueError("Tavily 搜索需要 API Key")
        body = json.dumps({"query": text, "max_results": limit, "search_depth": "basic",
                           "include_answer": False, "include_raw_content": False,
                           "include_images": False, "include_favicon": False,
                           **({"time_range": window} if window else {})}).encode()
        request = urllib.request.Request(base, data=body, headers={
            "Authorization": "Bearer " + str(api_key).strip(), "Content-Type": "application/json",
            "Accept": "application/json", "User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=max(1, min(30, timeout))) as response:
            payload = json.load(response)
        rows = [{"title": str(item.get("title") or "").strip(), "url": str(item.get("url") or "").strip(),
                 "snippet": str(item.get("content") or "").strip(),
                 "source": urllib.parse.urlparse(str(item.get("url") or "")).hostname or "",
                 "published_at": str(item.get("published_date") or "").strip()}
                for item in payload.get("results", [])]
    else:
        raise ValueError(f"未知搜索提供方：{provider}")
    rows = [row for row in rows if row["url"]][:limit]
    return {"provider": provider, "query": text, "results": rows,
            "answer": str(payload.get("answer") or "").strip() if provider == "tavily" else ""}
