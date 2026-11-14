import heapq
import math
import re
import time
from collections import Counter, defaultdict

from .dataset import evidence_groups
from .policy import BASELINE, gate_matches, hash_value, normalize_config

# This default is frozen so historical v0.1 numerical results remain reproducible.
STOP = set("a an and are as at be been but by can for from had has have how if in into is it its may more no not of on or our than that the their these this those to using was we were what when where which while who will with without would study studies results result patients patient showed show significantly significant".split())
NEGATIONS = {"no", "not", "without", "neither", "nor", "never", "absence", "lack", "lacking"}


def tokens(text, preserve=False):
    pattern = r"\d+(?:\.\d+)?%?|[a-z0-9]+(?:[-'][a-z0-9]+)*" if preserve else r"[a-z0-9]+(?:[-'][a-z0-9]+)*"
    stop = STOP - NEGATIONS if preserve else STOP
    return [w for w in re.findall(pattern, text.lower()) if w not in stop and
            (len(w) > 1 or (preserve and w.rstrip("%").replace(".", "", 1).isdigit()))]


class Retriever:
    def __init__(self, corpus):
        self.corpus_hash = hash_value(corpus)
        self.sentences, self.postings, self.idf = [], defaultdict(list), {}
        self.aliases = defaultdict(set)
        self.protected_postings, self.protected_lengths, self.protected_terms = defaultdict(list), [], []
        for doc in corpus:
            text = " ".join(doc["abstract"])
            for match in re.finditer(r"([A-Za-z][A-Za-z -]{3,100})\s*\(([A-Z][A-Z0-9-]{1,9})\)", text):
                acronym, words = match.group(2), match.group(1).strip().split()
                letters = re.sub(r"[^A-Z]", "", acronym)
                for n in range(2, min(len(words), len(letters) + 2) + 1):
                    phrase = words[-n:]
                    if "".join(w[0].upper() for w in phrase) == letters:
                        self.aliases[acronym.lower()].add(" ".join(phrase).lower())
                        break
            for index, sentence in enumerate(doc["abstract"]):
                text = doc["title"] + " " + sentence
                words, protected = tokens(text), tokens(text, preserve=True)
                sid = len(self.sentences)
                self.sentences.append({"id": f"{doc['doc_id']}:{index}", "doc_id": str(doc["doc_id"]),
                                       "sentence_index": index, "title": doc["title"], "text": sentence,
                                       "length": max(1, len(words)), "terms": words})
                self.protected_lengths.append(max(1, len(protected)))
                self.protected_terms.append(protected)
                for word, count in Counter(words).items():
                    self.postings[word].append((sid, count))
                for word, count in Counter(protected).items():
                    self.protected_postings[word].append((sid, count))
        size = len(self.sentences)
        self.average_length = sum(s["length"] for s in self.sentences) / max(1, size)
        self.protected_average = sum(self.protected_lengths) / max(1, size)
        self.idf = {w: math.log(1 + (size - len(p) + .5) / (len(p) + .5)) for w, p in self.postings.items()}
        self.protected_idf = {w: math.log(1 + (size - len(p) + .5) / (len(p) + .5)) for w, p in self.protected_postings.items()}
        self.by_id = {s["id"]: s for s in self.sentences}
        self.legacy_terms = [s["terms"] for s in self.sentences]

    def features(self, query):
        words = tokens(query)
        acronyms = sorted({w.lower() for w in re.findall(r"\b[A-Z][A-Z0-9-]{1,9}\b", query) if w.lower() in self.aliases})
        unknown = sum(w not in self.postings for w in words)
        return {"has_acronym": bool(acronyms),
                "unique_acronym": bool(acronyms) and all(len(self.aliases[a]) == 1 for a in acronyms),
                "ambiguous_acronym": any(len(self.aliases[a]) > 1 for a in acronyms),
                "short_query": len(words) <= 8, "query_terms": len(words), "acronyms": acronyms,
                "has_negation": bool(set(re.findall(r"[a-z]+", query.lower())) & NEGATIONS),
                "has_number": bool(re.search(r"\d", query)),
                "rare_terms": bool(words) and unknown / len(words) >= .2,
                "unknown_term_fraction": round(unknown / max(1, len(words)), 4)}

    def bm25(self, words, limit=20, preserve=False):
        scores = defaultdict(float)
        postings, idf = (self.protected_postings, self.protected_idf) if preserve else (self.postings, self.idf)
        average = self.protected_average if preserve else self.average_length
        for word in set(words):
            for sid, count in postings.get(word, []):
                length = self.protected_lengths[sid] if preserve else self.sentences[sid]["length"]
                scores[sid] += idf[word] * count * 2.5 / (count + 1.5 * (.25 + .75 * length / average))
        return heapq.nsmallest(limit, scores.items(), key=lambda pair: (-pair[1], pair[0]))

    def policy_triggers(self, query, config):
        cfg, f = normalize_config(config), self.features(query)
        if not gate_matches(cfg["gate"], f):
            return False
        return ((cfg["normalization"] == "preserve_constraints" and (f["has_negation"] or f["has_number"]))
                or cfg["strategy"] == "prf" or (cfg["strategy"] == "acronym" and f["has_acronym"]))

    def search(self, query, config, k=5):
        config = normalize_config(config)
        features = self.features(query)
        enabled = gate_matches(config["gate"], features)
        preserve = enabled and config["normalization"] == "preserve_constraints" and (features["has_negation"] or features["has_number"])
        original = tokens(query, preserve=preserve)
        first = self.bm25(original, preserve=preserve)
        candidates50 = self.bm25(original, limit=50, preserve=preserve)
        expansion, sources = [], []
        if enabled and config["strategy"] == "acronym":
            for acronym in features["acronyms"]:
                for phrase in sorted(self.aliases[acronym]):
                    expansion.extend(tokens(phrase, preserve=preserve))
                    sources.append({"kind": "corpus_glossary", "acronym": acronym, "phrase": phrase})
        if enabled and config["strategy"] == "prf":
            term_lists = self.protected_terms if preserve else self.legacy_terms
            idf = self.protected_idf if preserve else self.idf
            counts = Counter(word for sid, _ in first[:3] for word in term_lists[sid]
                             if word not in original and not word.rstrip("%").replace(".", "", 1).isdigit())
            expansion = sorted(counts, key=lambda w: (-counts[w] * idf.get(w, 0), w))[:config["prf_terms"]]
            sources = [{"kind": "pseudo_feedback", "sentence_id": self.sentences[sid]["id"]} for sid, _ in first[:3]]
        if expansion:
            second = self.bm25(original + expansion, preserve=preserve)
            fused = defaultdict(float)
            for ranking in (first, second):
                for rank, (sid, _) in enumerate(ranking, 1):
                    fused[sid] += 1 / (60 + rank)
            ranking = sorted(fused.items(), key=lambda p: (-p[1], p[0]))[:k]
        else:
            ranking = first[:k]
        hits = [{key: value for key, value in self.sentences[sid].items() if key not in ("terms", "length")} | {"score": round(score, 6)} for sid, score in ranking]
        return {"hits": hits, "expansion": sorted(set(expansion)), "sources": sources, "features": features,
                "initial_candidate_ids_50": [self.sentences[sid]["id"] for sid, _ in candidates50],
                "normalization_applied": "preserve_constraints" if preserve else "legacy"}


def score_claim(claim, result):
    groups = evidence_groups(claim)
    returned = {hit["id"] for hit in result["hits"]}
    coverage = max((len(returned & group) / len(group) for group in groups), default=0)
    complete = int(any(group <= returned for group in groups))
    relevant = set().union(*groups) if groups else set()
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(result["hits"]) if h["id"] in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
    return {"complete_at_5": complete, "coverage_at_5": coverage, "ndcg_at_5": dcg / ideal if ideal else 0}


def guard_accepts(baseline, candidate):
    return all(candidate[key] >= baseline[key] - 1e-12 for key in ("complete_at_5", "coverage_at_5"))


def aggregate(rows):
    return {key: sum(r[key] for r in rows) / len(rows) if rows else 0 for key in ("complete_at_5", "coverage_at_5", "ndcg_at_5")}


def subset_result(result, ids):
    wanted = set(map(str, ids))
    rows = [r for r in result["rows"] if r["claim_id"] in wanted]
    return {**result, "rows": rows, "count": len(rows), "metrics": aggregate(rows), "seconds": 0, "computed_rows": 0, "reused_rows": len(rows)}


def evaluate(retriever, claims, config, previous=None):
    start, rows = time.monotonic(), []
    config_key = hash_value(normalize_config(config))
    reusable = {}
    if previous and previous.get("config_hash") == config_key and previous.get("corpus_hash") == retriever.corpus_hash:
        reusable = {r["claim_id"]: r for r in previous["rows"]}
    reused = 0
    for claim in claims:
        key = str(claim["id"])
        if key in reusable and reusable[key].get("claim_hash") == hash_value(claim):
            rows.append(reusable[key]); reused += 1
            continue
        result = retriever.search(claim["claim"], config)
        groups = evidence_groups(claim)
        score = score_claim(claim, result)
        candidates = set(result["initial_candidate_ids_50"])
        failure = ("none" if score["complete_at_5"] else "ranking" if any(g <= candidates for g in groups)
                   else "partial_recall" if any(g & candidates for g in groups) else "recall")
        rows.append({"claim_id": key, "claim": claim["claim"], **result, **score,
                     "gold_sets": [sorted(g) for g in groups], "failure_type": failure,
                     "claim_hash": hash_value(claim)})
    return {"metrics": aggregate(rows), "count": len(rows), "seconds": time.monotonic() - start,
            "rows": rows, "computed_rows": len(rows) - reused, "reused_rows": reused,
            "config_hash": config_key, "corpus_hash": retriever.corpus_hash}
