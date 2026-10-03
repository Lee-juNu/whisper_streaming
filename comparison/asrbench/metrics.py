"""CER, nickname and percentile metrics (stdlib only)."""
import math
import unicodedata

NORMALIZATION = {
    "name": "nfkc_casefold_strip_ws_punct_v1",
    "steps": [
        "unicodedata.normalize('NFKC', text)",
        "str.casefold()",
        "remove whitespace (str.isspace() or Unicode category Z*)",
        "remove Unicode punctuation (Unicode category P*)",
    ],
    "preserved": "kana / kanji / digits / latin / symbols (S*) are kept as-is; no kana<->kanji, "
                 "reading, numeral or concept rewriting. Raw transcripts are retained separately.",
}

PERCENTILE_METHOD = "nearest-rank: sorted ascending, rank = ceil(p/100 * n) (1-based), no interpolation"


def normalize(text):
    if text is None:
        return None
    t = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in t if not (ch.isspace() or unicodedata.category(ch)[0] in "ZP"))


def edit_counts(ref, hyp):
    """Levenshtein alignment counts between two (already normalized) strings."""
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]  # (cost, sub, del, ins)
    for i, rc in enumerate(ref, 1):
        cur = [(i, 0, i, 0)]
        for j, hc in enumerate(hyp, 1):
            if rc == hc:
                cur.append(prev[j - 1])
                continue
            s, d, n = prev[j - 1], prev[j], cur[j - 1]
            cur.append(min((s[0] + 1, s[1] + 1, s[2], s[3]),
                           (d[0] + 1, d[1], d[2] + 1, d[3]),
                           (n[0] + 1, n[1], n[2], n[3] + 1)))
        prev = cur
    cost, sub, dele, ins = prev[-1]
    return {"substitutions": sub, "deletions": dele, "insertions": ins, "edits": cost,
            "ref_chars": len(ref), "hyp_chars": len(hyp)}


def utterance_cer(ref, hyp):
    """Per-utterance CER detail; None when there is no reference or no hypothesis."""
    if ref is None or hyp is None:
        return None
    r, h = normalize(ref), normalize(hyp)
    c = edit_counts(r, h)
    c["cer"] = c["edits"] / c["ref_chars"] if c["ref_chars"] else None
    c["ref_norm"], c["hyp_norm"] = r, h
    return c


def cer_aggregate(pairs):
    """Micro CER over (reference, hypothesis) raw-text pairs.

    reference None  -> no reference: excluded, counted (no CER is inferred).
    hypothesis None -> failure: excluded from micro_cer, reported, and also folded in
                       as full deletions in micro_cer_failures_as_deletions.
    empty normalized reference -> excluded from the denominator; insertions reported.
    """
    a = dict(utterances=0, no_reference=0, failed_with_reference=0, failed_ref_chars=0,
             empty_reference=0, empty_reference_insertions=0, scored=0, ref_chars=0,
             substitutions=0, deletions=0, insertions=0)
    for ref, hyp in pairs:
        a["utterances"] += 1
        if ref is None:
            a["no_reference"] += 1
            continue
        r = normalize(ref)
        if hyp is None:
            a["failed_with_reference"] += 1
            a["failed_ref_chars"] += len(r)
            continue
        c = edit_counts(r, normalize(hyp))
        if not r:
            a["empty_reference"] += 1
            a["empty_reference_insertions"] += c["insertions"]
            continue
        a["scored"] += 1
        for k in ("ref_chars", "substitutions", "deletions", "insertions"):
            a[k] += c[k]
    a["edits"] = a["substitutions"] + a["deletions"] + a["insertions"]
    a["micro_cer"] = a["edits"] / a["ref_chars"] if a["ref_chars"] else None
    denom = a["ref_chars"] + a["failed_ref_chars"]
    a["micro_cer_failures_as_deletions"] = (a["edits"] + a["failed_ref_chars"]) / denom if denom else None
    return a


def mention_counts(text, names):
    """Count canonical-name mentions in text: normalized, left-to-right, longest match first,
    non-overlapping (so "タロウ" consumes its inner "タロ")."""
    by_norm = {}
    for n in names:
        k = normalize(n)
        if k:
            by_norm.setdefault(k, n)
    keys = sorted(by_norm, key=lambda k: (-len(k), k))
    counts = {n: 0 for n in names}
    t = normalize(text or "")
    i = 0
    while i < len(t):
        for k in keys:
            if t.startswith(k, i):
                counts[by_norm[k]] += 1
                i += len(k)
                break
        else:
            i += 1
    return counts


def nickname_utterance(hyp, candidates, spoken):
    """Exact canonical-spelling nickname scoring against manually counted mentions.

    The false-insertion universe is limited to candidates + spoken names; insertions of any
    other name are invisible to this metric. spoken None -> not annotated -> None.
    hyp None -> failed utterance (reported, not aggregated).
    """
    if spoken is None:
        return None
    universe = list(dict.fromkeys(list(candidates) + list(spoken)))
    got = mention_counts(hyp, universe) if hyp is not None else {n: 0 for n in universe}
    per = {}
    tot = dict(spoken_mentions=0, hits=0, misses=0, false_insertions=0, distractor_false_insertions=0)
    for n in universe:
        r, h = spoken.get(n, 0), got[n]
        hit = min(r, h)
        row = {"spoken": r, "hypothesis": h, "hits": hit, "misses": r - hit,
               "false_insertions": max(h - r, 0), "role": "spoken" if r else "distractor"}
        per[n] = row
        tot["spoken_mentions"] += r
        tot["hits"] += hit
        tot["misses"] += r - hit
        tot["false_insertions"] += row["false_insertions"]
        if not r:
            tot["distractor_false_insertions"] += row["false_insertions"]
    return {"failed": hyp is None, "per_name": per, **tot}


def nickname_aggregate(rows):
    agg = dict(utterances_scored=0, utterances_failed=0, utterances_with_false_insertion=0,
               spoken_mentions=0, hits=0, misses=0, false_insertions=0, distractor_false_insertions=0)
    per_name = {}
    for row in rows:
        if row["failed"]:
            agg["utterances_failed"] += 1
            continue
        agg["utterances_scored"] += 1
        agg["utterances_with_false_insertion"] += row["false_insertions"] > 0
        for k in ("spoken_mentions", "hits", "misses", "false_insertions", "distractor_false_insertions"):
            agg[k] += row[k]
        for n, r in row["per_name"].items():
            p = per_name.setdefault(n, {"spoken": 0, "hits": 0, "misses": 0, "false_insertions": 0})
            for k in p:
                p[k] += r[k]
    agg["mention_recall"] = agg["hits"] / agg["spoken_mentions"] if agg["spoken_mentions"] else None
    found = agg["hits"] + agg["false_insertions"]
    agg["candidate_list_precision"] = agg["hits"] / found if found else None
    agg["per_name"] = per_name
    agg["definition"] = ("exact canonical spelling after CER normalization; longest-match non-overlapping "
                         "counts vs manually counted spoken mentions; false insertions only measurable for "
                         "names in the candidate list (incl. distractors). Not a general proper-noun error rate.")
    return agg


def nearest_rank(sorted_values, p):
    k = max(1, math.ceil(p / 100.0 * len(sorted_values)))
    return sorted_values[k - 1]


def describe(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None,
                "method": PERCENTILE_METHOD}
    return {"n": len(v), "min": v[0], "p50": nearest_rank(v, 50), "p95": nearest_rank(v, 95),
            "max": v[-1], "mean": sum(v) / len(v), "method": PERCENTILE_METHOD}
