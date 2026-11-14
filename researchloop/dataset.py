import hashlib
import io
import json
import random
import tarfile
import urllib.request
from collections import defaultdict

from .storage import DATA, ROOT, fingerprint, now, read_json, write_json

SOURCE = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
LICENSE = "https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "ResearchLoop/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load_lines(name):
    with (DATA / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_groups(claim):
    return [frozenset(f"{doc}:{s}" for s in item["sentences"])
            for doc, rationales in claim.get("evidence", {}).items()
            for item in rationales if item.get("sentences")]


def make_splits(train, dev, seed=2026, target=60, test_size=80):
    """Group claims by shared gold documents; no component crosses a partition."""
    rows = {str(c["id"]): c for c in train + dev if evidence_groups(c)}
    parent = {cid: cid for cid in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    docs = {}
    for cid, row in rows.items():
        for doc in row["evidence"]:
            if doc in docs:
                parent[find(cid)] = find(docs[doc])
            docs[doc] = cid
    groups = defaultdict(list)
    for cid in rows:
        groups[find(cid)].append(cid)
    train_ids = {str(c["id"]) for c in train}
    dev_ids = {str(c["id"]) for c in dev}
    rng = random.Random(seed)
    components = list(groups.values())
    rng.shuffle(components)

    def take(pool, allowed, count):
        chosen, remaining = [], []
        for group in pool:
            eligible = [cid for cid in group if cid in allowed]
            if eligible and len(chosen) + len(eligible) <= count:
                chosen.extend(eligible)
            else:
                remaining.append(group)
        return chosen, remaining

    test, components = take(components, dev_ids, test_size)
    a_explore, components = take(components, train_ids, target * 2 // 3)
    a_guard, components = take(components, train_ids, target - target * 2 // 3)
    b_explore, components = take(components, train_ids, target * 2 // 3)
    b_guard, _ = take(components, train_ids, target - target * 2 // 3)
    splits = {"A": {"explore": a_explore, "guard": a_guard},
              "B": {"explore": b_explore, "guard": b_guard}, "test": test}
    parts = [a_explore, a_guard, b_explore, b_guard, test]
    doc_sets = [{d for cid in part for d in rows[cid]["evidence"]} for part in parts]
    for i, left in enumerate(doc_sets):
        for right in doc_sets[i + 1:]:
            assert not left & right, "Gold document leakage"
    if min(map(len, parts)) < 5:
        raise ValueError("分组后数据不足，请检查原始数据")
    return splits


def prepare(progress=lambda _: None):
    DATA.mkdir(parents=True, exist_ok=True)
    archive_path = DATA / "source.tar.gz"
    if not archive_path.exists():
        progress("下载 SciFact 原始数据（约 3 MB）")
        raw = fetch(SOURCE)
        archive_path.write_bytes(raw)
    else:
        raw = archive_path.read_bytes()
    names = {"corpus.jsonl", "claims_train.jsonl", "claims_dev.jsonl"}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.rsplit("/", 1)[-1]
            if name in names and member.isfile():
                (DATA / name).write_bytes(archive.extractfile(member).read())
    if not all((DATA / n).exists() for n in names):
        raise ValueError("原始归档缺少必要文件")
    license_path = DATA / "LICENSE.md"
    if not license_path.exists():
        license_path.write_bytes(fetch(LICENSE))
    progress("按共享证据文档分组，冻结探索、验收和最终测试划分")
    train, dev = load_lines("claims_train.jsonl"), load_lines("claims_dev.jsonl")
    splits = make_splits(train, dev)
    manifest = {"name": "SciFact", "source": SOURCE, "downloaded_at": now(),
                "archive_sha256": hashlib.sha256(raw).hexdigest(), "split_seed": 2026,
                "licenses": {"annotations": "CC BY 4.0", "abstracts": "ODC-By 1.0"},
                "license_source": "https://github.com/allenai/scifact/blob/master/LICENSE.md",
                "corpus_count": len(load_lines("corpus.jsonl")),
                "split_policy": "Shared gold-document components are disjoint across all five partitions; evidence-positive claims only.",
                "splits": splits,
                "file_hashes": {n: hashlib.sha256((DATA / n).read_bytes()).hexdigest() for n in sorted(names)}}
    write_json(DATA / "manifest.json", manifest)
    progress("数据准备完成")
    return manifest


class Dataset:
    def __init__(self):
        self.manifest = read_json(DATA / "manifest.json")
        if not self.manifest:
            raise ValueError("请先准备 SciFact 数据")
        for name, expected in self.manifest["file_hashes"].items():
            if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"数据文件被修改：{name}")
        self.corpus = load_lines("corpus.jsonl")
        self.claims = {str(c["id"]): c for c in load_lines("claims_train.jsonl") + load_lines("claims_dev.jsonl")}
        self.registry = ensure_research_tasks(self.manifest, load_lines("claims_train.jsonl"), load_lines("claims_dev.jsonl"))
        self.tasks = {k: v for k, v in self.manifest["splits"].items() if k != "test"} | self.registry["tasks"]

    def task(self, name, partition):
        if name not in self.tasks or partition not in ("explore", "guard", "screen"):
            raise ValueError("无效任务划分")
        ids = self.tasks[name].get(partition, self.tasks[name]["explore"][:8])
        return [self.claims[cid] for cid in ids]


def ensure_research_tasks(manifest, train, dev, seed=20261114):
    """Freeze new tasks by IDs and gold-document components, never by performance."""
    path = DATA / "research_tasks_v2.json"
    existing = read_json(path)
    if existing:
        if existing["archive_sha256"] != manifest["archive_sha256"]:
            raise ValueError("新任务注册表与当前语料哈希不一致")
        return existing
    rows = {str(c["id"]): c for c in train + dev if evidence_groups(c)}
    parents = {cid: cid for cid in rows}
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    seen = {}
    for cid, row in rows.items():
        for doc in row["evidence"]:
            if doc in seen:
                parents[find(cid)] = find(seen[doc])
            seen[doc] = cid
    components = defaultdict(list)
    for cid in rows:
        components[find(cid)].append(cid)
    excluded_ids = set(manifest["splits"]["test"])
    for name in ("A", "B"):
        for ids in manifest["splits"][name].values():
            excluded_ids.update(ids)
    train_ids = {str(c["id"]) for c in train}
    pool = [[cid for cid in group if cid in train_ids] for group in components.values() if not set(group) & excluded_ids]
    pool = [g for g in pool if g]
    rng = random.Random(seed)
    rng.shuffle(pool)
    def take(count):
        nonlocal pool
        chosen, remaining = [], []
        for group in pool:
            if len(chosen) + len(group) <= count:
                chosen.extend(group)
            else:
                remaining.append(group)
        pool = remaining
        if len(chosen) < count:
            raise ValueError("隔离旧任务与最终测试后，新任务数量不足")
        return chosen
    tasks = {}
    for name in ("C", "D", "E", "F"):
        explore, guard = take(24), take(12)
        tasks[name] = {"explore": explore, "guard": guard,
                       "screen": sorted(explore, key=lambda cid: fingerprint({"seed": seed, "id": cid}))[:8]}
    registry = {"schema_version": 2, "created_at": now(), "seed": seed,
                "archive_sha256": manifest["archive_sha256"], "tasks": tasks,
                "method": "Training claims only; exclude entire components sharing old A/B or final-test gold documents; no retrieval results used.",
                "registry_hash": fingerprint(tasks)}
    write_json(path, registry)
    write_json(ROOT / "docs" / "experiments" / "v2-task-manifest.json", registry)
    return registry
