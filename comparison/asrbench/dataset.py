"""JSONL dataset loading/validation. Truth (reference, spoken nicknames, manual speech times)
is kept in ``Item.truth`` and is never part of ``adapters.RecognitionInput``."""
import json
from dataclasses import dataclass
from pathlib import Path

from . import audio
from .guards import find_placeholders
from .metrics import mention_counts, normalize

REQUIRED = {
    "utt_id": str, "session_id": str, "order": int, "audio_path": str,
    "reference": (str, type(None)), "reference_verified": bool,
    "speech_start_ms": int, "speech_end_ms": int,
    "nickname_candidates": list, "spoken_nicknames": (dict, type(None)),
}
OPTIONAL = {"synthetic": bool, "reference_kind": str, "notes": str}


@dataclass(frozen=True)
class Truth:
    reference: object           # str | None (None = no verified transcript -> no CER)
    reference_verified: bool
    speech_start_ms: int        # manual, relative to WAV sample 0 (= replay origin)
    speech_end_ms: int
    spoken_nicknames: object    # {canonical: mention count} | None (= not annotated)


@dataclass(frozen=True)
class Item:
    utt_id: str
    session_id: str
    order: int
    audio_path: Path
    nickname_candidates: tuple  # bias list / known universe incl. distractors (may be fed to models)
    synthetic: bool
    truth: Truth                # scoring only, never fed to models
    line: int


def has_errors(issues):
    return any(i["level"] == "error" for i in issues)


def _type_ok(v, t):
    if t is int or (isinstance(t, tuple) and int in t):
        if isinstance(v, bool):
            return False
    return isinstance(v, t)


def load_dataset(path, check_audio=True, allow_placeholders=False):
    path = Path(path)
    issues, items = [], []
    seen_ids, seen_order = set(), set()

    def issue(level, line, utt, msg):
        issues.append({"level": level, "line": line, "utt_id": utt, "message": msg})

    with open(path, encoding="utf-8") as f:
        lines = list(enumerate(f, 1))
    for ln, raw in lines:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as e:
            issue("error", ln, None, f"invalid JSON: {e}")
            continue
        if not isinstance(obj, dict):
            issue("error", ln, None, "line must be a JSON object")
            continue
        utt = obj.get("utt_id")
        bad = False
        for k, t in REQUIRED.items():
            if k not in obj:
                issue("error", ln, utt, f"missing required field {k!r} (use null where nullable)")
                bad = True
            elif not _type_ok(obj[k], t):
                issue("error", ln, utt, f"field {k!r} has wrong type {type(obj[k]).__name__}")
                bad = True
        for k, t in OPTIONAL.items():
            if k in obj and not _type_ok(obj[k], t):
                issue("error", ln, utt, f"field {k!r} has wrong type")
                bad = True
        for k in set(obj) - set(REQUIRED) - set(OPTIONAL):
            issue("warning", ln, utt, f"unknown field {k!r} ignored")
        if bad:
            continue
        synthetic = obj.get("synthetic", False)
        if not utt or not obj["session_id"]:
            issue("error", ln, utt, "utt_id/session_id must be non-empty")
            continue
        if utt in seen_ids:
            issue("error", ln, utt, "duplicate utt_id")
            continue
        seen_ids.add(utt)
        key = (obj["session_id"], obj["order"])
        if key in seen_order:
            issue("error", ln, utt, f"duplicate (session_id, order) {key}")
            continue
        seen_order.add(key)
        s, e = obj["speech_start_ms"], obj["speech_end_ms"]
        if not 0 <= s < e:
            issue("error", ln, utt, "need 0 <= speech_start_ms < speech_end_ms")
            continue
        ref = obj["reference"]
        if ref is not None and not obj["reference_verified"]:
            if synthetic:
                issue("warning", ln, utt, "synthetic placeholder reference (not a real transcript)")
            else:
                issue("error", ln, utt, "reference present but reference_verified=false; verify it or set null")
                continue
        cands = obj["nickname_candidates"]
        if not all(isinstance(c, str) and c.strip() for c in cands):
            issue("error", ln, utt, "nickname_candidates must be non-empty strings")
            continue
        norms = [normalize(c) for c in cands]
        if len(set(norms)) != len(norms) or not all(norms):
            issue("error", ln, utt, "nickname_candidates collide or vanish after normalization")
            continue
        spoken = obj["spoken_nicknames"]
        if spoken is not None:
            if not all(isinstance(k, str) and normalize(k) and _type_ok(v, int) and v >= 1
                       for k, v in spoken.items()):
                issue("error", ln, utt, "spoken_nicknames must map canonical name -> mention count >= 1")
                continue
            for k in spoken:
                if k not in cands:
                    issue("warning", ln, utt, f"spoken nickname {k!r} is not in nickname_candidates "
                                              "(still scored; bias cannot include it)")
            if ref is not None:
                counted = mention_counts(ref, list(dict.fromkeys(list(cands) + list(spoken))))
                diff = {k: v for k, v in counted.items() if v != spoken.get(k, 0)}
                if diff:
                    issue("warning", ln, utt, f"longest-match counts in reference {diff} differ from "
                                              "spoken_nicknames; re-check annotation")
        ap = obj["audio_path"]
        if find_placeholders(ap):
            issue("warning" if allow_placeholders else "error", ln, utt,
                  f"audio_path is a placeholder ({ap}); not runnable")
            continue
        apath = Path(ap) if Path(ap).is_absolute() else (path.parent / ap)
        if check_audio:
            try:
                info = audio.wav_info(apath)
            except FileNotFoundError:
                issue("error", ln, utt, f"audio not found: {apath}")
                continue
            except (audio.WavFormatError, EOFError, OSError) as ex:
                issue("error", ln, utt, str(ex))
                continue
            if e > info["duration_ms"]:
                issue("error", ln, utt, f"speech_end_ms {e} beyond audio duration {info['duration_ms']:.0f}ms")
                continue
        items.append(Item(utt, obj["session_id"], obj["order"], apath, tuple(cands), synthetic,
                          Truth(ref, obj["reference_verified"], s, e,
                                dict(spoken) if spoken is not None else None), ln))
    first = {}
    for it in items:
        first.setdefault(it.session_id, len(first))
    items.sort(key=lambda it: (first[it.session_id], it.order))
    return items, issues
