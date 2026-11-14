"""Platform source adapters for research evidence.

Public endpoints are used where they exist (Bilibili search). Platforms that
need a login or are anti-bot protected (Xiaohongshu, Weibo, Douyin, Zhihu) are
reached through a site-restricted web search provider; logged-in content is
never scraped. Each result records the platform and a clear evidence scope.
"""
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

PLATFORMS = {
    "bilibili": {"label": "哔哩哔哩", "site": "bilibili.com", "open_api": True},
    "xiaohongshu": {"label": "小红书", "site": "xiaohongshu.com", "open_api": False},
    "tieba": {"label": "百度贴吧", "site": "tieba.baidu.com", "open_api": False},
    "weibo": {"label": "微博", "site": "weibo.com", "open_api": False},
    "douyin": {"label": "抖音", "site": "douyin.com", "open_api": False},
    "zhihu": {"label": "知乎", "site": "zhihu.com", "open_api": False},
}


def describe():
    return [{"platform": name, "label": value["label"], "open_api": value["open_api"],
             "site": value["site"]} for name, value in PLATFORMS.items()]


def _clean(text):
    return html.unescape(re.sub(r"<[^>]+>", "", str(text or ""))).strip()


def _bilibili(query, limit, timeout=10):
    url = "https://api.bilibili.com/x/web-interface/search/type?" + urllib.parse.urlencode(
        {"search_type": "video", "keyword": query, "page": 1})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Referer": "https://www.bilibili.com/",
                                                   "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=max(1, min(20, timeout))) as response:
        payload = json.load(response)
    if payload.get("code") not in (0, None):
        raise ValueError(f"Bilibili 检索失败：code={payload.get('code')} {payload.get('message') or ''}")
    items = (payload.get("data") or {}).get("result") or []
    results = []
    for item in items[:max(1, min(int(limit or 5), 10))]:
        bvid = str(item.get("bvid") or "").strip()
        if not bvid:
            continue
        published = item.get("pubdate")
        results.append({
            "platform": "bilibili", "title": _clean(item.get("title")),
            "url": f"https://www.bilibili.com/video/{bvid}",
            "snippet": _clean(item.get("description"))[:200],
            "author": _clean(item.get("author")),
            "metrics": {"play": item.get("play"), "danmaku": item.get("video_review")},
            "published_at": datetime.fromtimestamp(int(published), tz=timezone.utc).isoformat()
            if isinstance(published, int) and published > 0 else ""})
    return results


def _tieba(query, limit, timeout=10):
    url = "https://tieba.baidu.com/f/search/res?" + urllib.parse.urlencode({"ie": "utf-8", "qw": query})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    with urllib.request.urlopen(request, timeout=max(1, min(20, timeout))) as response:
        body = response.read(1 << 20).decode("utf-8", errors="replace")
    links = re.findall(r'href="(/p/\d+)"[^>]*>(.*?)</a>', body, re.S)
    results = []
    seen = set()
    for path, raw in links:
        title = _clean(raw)
        if not title or path in seen:
            continue
        seen.add(path)
        results.append({"platform": "tieba", "title": title[:120],
                        "url": "https://tieba.baidu.com" + path, "snippet": "",
                        "author": "", "metrics": {}, "published_at": ""})
        if len(results) >= max(1, min(int(limit or 5), 10)):
            break
    if not results:
        raise ValueError("贴吧检索未返回可解析结果（可能需要登录或触发反爬），未编造贴文内容")
    return results


def _site_search(platform, query, limit, web):
    site = PLATFORMS[platform]["site"]
    provider = str(web.get("search_provider") or "searxng")
    endpoint = str(web.get("search_endpoint") or "").strip()
    if not web.get("enabled") or not endpoint:
        raise ValueError(f"{PLATFORMS[platform]['label']}没有公开检索接口；"
                         f"请在设置中启用联网搜索（SearXNG/Tavily），系统会用 site:{site} 限定检索")
    from . import webaccess
    payload = webaccess.search(f"site:{site} {query}", provider=provider, endpoint=endpoint,
                               api_key=web.get("search_api_key"), max_results=limit, recency="any")
    results = []
    for row in payload.get("results", []):
        results.append({"platform": platform, "title": row.get("title", ""), "url": row.get("url", ""),
                        "snippet": row.get("snippet", ""), "author": row.get("source", ""),
                        "metrics": {}, "published_at": row.get("published_at", "")})
    return results


def _search(platform, query, limit=5, timeout=10, blocked_resources=()):
    from . import storage
    name = str(platform or "").strip().lower()
    text = str(query or "").strip()
    if not text:
        raise ValueError("平台检索需要关键词")
    if name not in PLATFORMS:
        raise ValueError(f"未知平台 {name}；可用：{'、'.join(PLATFORMS)}")
    label = PLATFORMS[name]["label"]
    web = (storage.settings() or {}).get("web") or {}
    search_resource = "search:" + str(web.get("search_provider") or "searxng")
    if name == "bilibili":
        results = _bilibili(text, limit, timeout)
        note = "Bilibili 公开搜索接口；视频标题/播放量是平台展示值，不等于研究结论。"
    elif name == "tieba":
        try:
            results = _tieba(text, limit, timeout)
            note = "贴吧公开检索页；只解析帖子标题与链接，未采集帖内用户内容。"
        except Exception:
            if search_resource in blocked_resources:
                raise ValueError("贴吧直连失败，备用搜索服务当前配置已被停用；请改用其他来源")
            results = _site_search(name, text, limit, web)
            note = "贴吧直连被反爬拦截，改用站内限定搜索。"
    else:
        if search_resource in blocked_resources:
            from .reliability import Failure
            raise Failure("该平台依赖的搜索服务已鉴权失败，请改用可用来源或修复配置", resource=search_resource, blocked=True)
        results = _site_search(name, text, limit, web)
        note = f"{label}需要登录或反爬保护，未直接抓取平台内容；结果来自 site:{PLATFORMS[name]['site']} 的站内限定搜索。"
    if not results:
        raise ValueError(f"{label}检索没有返回结果（关键词、登录或反爬限制），未编造内容")
    return {"platform": name, "label": label, "query": text, "results": results[:max(1, min(int(limit or 5), 10))],
            "note": note}


def search(platform, query, limit=5, timeout=10, blocked_resources=()):
    from . import reliability
    try:
        return _search(platform, query, limit, timeout, blocked_resources)
    except Exception as error:
        raise reliability.wrap(error, "platform:" + str(platform)) from error
