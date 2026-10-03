"""Benchmark runner: file-controlled / replay-controlled / replay-native, raw event
preservation, scoring and unique (never overwritten) output directories.

Engines only ever receive ``RecognitionInput``; ``Item.truth`` is attached to the record
after recognition, for scoring."""
import json
import os
import platform
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import ROOT, __version__, audio, metrics
from .adapters import EV_COMPLETED, EV_DELTA, RecognitionInput

JST = timezone(timedelta(hours=9), "JST")
MODES = ("file-controlled", "replay-controlled", "replay-native")
COMPARABILITY = {
    "file-controlled": "whole WAV in one decode/request; only processing_ms is measured, latency fields are "
                       "null by design. Not comparable with replay latencies.",
    "replay-controlled": "same audio paced in real time; the SAME host energy endpointer (profile.endpoint) "
                         "decides when Whisper decodes / when Nemotron is committed (server endpointing "
                         "disabled). Final latency is comparable across engines within this mode only.",
    "replay-native": "Nemotron only: whole WAV paced, server-native endpointing emits finals; the EOF commit "
                     "only flushes. Different endpointer, so latency is NOT comparable with replay-controlled.",
}
LATENCY_DEFINITION = ("final_latency_ms = receive time of the last content-bearing final - (replay origin + "
                      "manual speech_end_ms); first_partial_latency_ms = first delta receive time - (replay "
                      "origin + manual speech_start_ms); negative final latency => early_final; empty finals "
                      "(e.g. EOF flush) are kept as raw finals but never replace content")
MOCK_LABEL = "MOCK: synthetic tones + scripted engine output; says nothing about model quality or speed"


def now_jst():
    return datetime.now(JST).isoformat(timespec="seconds")


def load_profile(name_or_path):
    p = Path(name_or_path)
    if p.suffix != ".json":
        p = ROOT / "profiles" / f"{name_or_path}.json"
    data = p.read_bytes()
    return json.loads(data), {"path": str(p), "sha256": audio.sha256_bytes(data)}


def load_lock(path=ROOT / "lock.json"):
    data = Path(path).read_bytes()
    return json.loads(data), audio.sha256_bytes(data)


def run_dir_name(profile_name, mode, bias, mock):
    ts = datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{profile_name}_{mode}_bias-{bias}{'_MOCK' if mock else ''}_{os.urandom(3).hex()}"


def make_run_dir(out_root, name):
    d = Path(out_root) / name
    d.parent.mkdir(parents=True, exist_ok=True)
    d.mkdir(exist_ok=False)  # refuse to overwrite an existing run
    return d


def make_input(item):
    return RecognitionInput(item.utt_id, item.session_id, item.audio_path, audio.sha256_file(item.audio_path),
                            audio.read_wav(item.audio_path), item.nickname_candidates)


def _ms(t, ref):
    return None if t is None or ref is None else (t - ref) * 1000.0


def _finals(events, commit_index):
    out = []
    for i, ev in enumerate(events):
        if ev["type"] == EV_COMPLETED:
            tr = ev["raw"].get("transcript") if isinstance(ev["raw"], dict) else None
            out.append({"event_index": i, "t": ev["t"], "transcript": tr, "empty": not (tr or "").strip(),
                        "after_commit": commit_index is not None and i >= commit_index})
    return out


def _stream(engine, inp, phrases, endpointing_ms, clock, packet, stop):
    sess = engine.open_stream(inp, phrases, endpointing_ms)
    replay = {}
    try:
        sess.start()
        t0, lag = audio.paced_replay(inp.samples, packet, clock,
                                     lambda chunk, end: sess.send_audio(audio.pcm16le(chunk)), stop)
        replay.update(origin_t=t0, max_send_lag_ms=lag, sent_samples=stop or len(inp.samples))
        sess.commit()
        replay["finish_ok"] = sess.finish()
        replay["commit_t"] = sess.commit_t
    except Exception as e:
        e.events = sess.events  # keep raw events of a failed stream
        raise
    finally:
        sess.close()
    return sess, t0, replay


def recognize(engine, profile, mode, item, inp, clock, phrases, prompt):
    """Run one utterance. Returns (record, raw events). Truth is used only for latency anchors
    AFTER recognition (speech_start/end_ms), never passed to the engine."""
    rec = {"finals": [], "hypothesis": None, "raw_text": None, "processing_ms": None, "replay": None,
           "segments": None, "endpoint_decision": None, "failure": None}
    events, t0 = [], None
    if mode == "file-controlled":
        out = engine.transcribe_file(inp, prompt, phrases)
        events.append({"t": clock.now(), "type": "harness.file_result", "raw": out})
        rec.update(hypothesis=out["text"], raw_text=out["raw_text"], processing_ms=out["processing_ms"])
        return rec, events, None
    cfg = audio.EndpointConfig(**profile["endpoint"])
    packet = audio.ms_to_samples(profile.get("replay", {}).get("packet_ms", 20))
    if mode == "replay-controlled":
        segs, short = audio.find_segments(inp.samples, cfg)
        seg, reason = audio.classify_single(segs)
        rec["segments"] = {"kept": [s.as_ms() for s in segs], "filtered_short": [s.as_ms() for s in short]}
        rec["endpoint_decision"] = reason or "single_segment"
        if reason == "no_segment_detected":
            rec["hypothesis"] = rec["raw_text"] = ""  # endpointer never fires -> nothing is transcribed
            return rec, events, None
        if reason:
            rec["failure"] = f"endpoint:{reason}"
            return rec, events, None
        if engine.kind == "faster_whisper":
            t0, lag = audio.paced_replay(inp.samples, packet, clock, lambda chunk, end: None, seg.endpoint)
            out = engine.decode_segment(inp, inp.samples[seg.start:seg.endpoint], prompt, phrases)
            t = clock.now()
            events.append({"t": t, "type": "harness.decode_result", "raw": out})
            rec["replay"] = {"origin_t": t0, "max_send_lag_ms": lag, "sent_samples": seg.endpoint}
            rec["finals"] = [{"event_index": 0, "t": t, "transcript": out["text"],
                              "empty": not out["text"].strip(), "after_commit": True}]
            rec.update(raw_text=out["raw_text"], processing_ms=out["processing_ms"])
        else:
            sess, t0, rec["replay"] = _stream(engine, inp, phrases, profile["session_endpointing_ms"], clock,
                                              packet, seg.endpoint)
            events = sess.events
            rec["finals"] = _finals(events, sess.commit_index)
    elif mode == "replay-native":
        sess, t0, rec["replay"] = _stream(engine, inp, phrases, profile["session_endpointing_ms"], clock,
                                          packet, None)
        events = sess.events
        rec["finals"] = _finals(events, sess.commit_index)
    else:
        raise ValueError(f"unknown mode {mode}")
    content = [f["transcript"] for f in rec["finals"] if not f["empty"]]
    if rec["finals"]:
        rec["hypothesis"] = "".join(content)
        if rec["raw_text"] is None:
            rec["raw_text"] = rec["hypothesis"]
    elif rec["failure"] is None:
        rec["failure"] = "no_final_received"
    return rec, events, t0


def add_latency(rec, truth, events, t0):
    end = None if t0 is None else t0 + truth.speech_end_ms / 1000.0
    start = None if t0 is None else t0 + truth.speech_start_ms / 1000.0
    for f in rec["finals"]:
        f["latency_ms"] = _ms(f["t"], end)
        f["early"] = f["latency_ms"] is not None and f["latency_ms"] < 0
    content = [f for f in rec["finals"] if not f["empty"]]
    rec["final_latency_ms"] = _ms(content[-1]["t"], end) if content else None
    rec["early_final"] = rec["final_latency_ms"] is not None and rec["final_latency_ms"] < 0
    rec["any_early_final"] = any(f["early"] for f in content)
    deltas = [e["t"] for e in events if e["type"] == EV_DELTA]
    rec["first_partial_latency_ms"] = _ms(deltas[0], start) if deltas else None
    rec["n_finals"], rec["n_content_finals"] = len(rec["finals"]), len(content)
    rec["multiple_finals"] = len(content) > 1


def score(records):
    per = []
    for r in records:
        t = r["truth"]
        per.append({"utt_id": r["utt_id"], "cer": metrics.utterance_cer(t["reference"], r["hypothesis"]),
                    "nickname": metrics.nickname_utterance(r["hypothesis"], r["nickname_candidates"],
                                                           t["spoken_nicknames"])})
    summary = {
        "n_utterances": len(records),
        "cer": metrics.cer_aggregate((r["truth"]["reference"], r["hypothesis"]) for r in records),
        "nickname": metrics.nickname_aggregate(p["nickname"] for p in per if p["nickname"] is not None),
        "final_latency_ms": metrics.describe(r.get("final_latency_ms") for r in records),
        "first_partial_latency_ms": metrics.describe(r.get("first_partial_latency_ms") for r in records),
        "processing_ms": metrics.describe(r.get("processing_ms") for r in records),
        "early_final_utterances": [r["utt_id"] for r in records if r.get("early_final")],
        "any_early_final_utterances": [r["utt_id"] for r in records if r.get("any_early_final")],
        "multiple_final_utterances": [r["utt_id"] for r in records if r.get("multiple_finals")],
        "failures": [{"utt_id": r["utt_id"], "failure": r["failure"]} for r in records if r["failure"]],
        "normalization": metrics.NORMALIZATION,
    }
    return per, summary


def _jdump(obj):
    return json.dumps(obj, ensure_ascii=False, default=str)


def run_benchmark(profile, profile_meta, mode, items, engine, clock, out_root, bias="off",
                  extra_meta=None, dataset_path=None, vram_summary=None):
    if mode not in profile["modes"]:
        raise ValueError(f"profile {profile['name']} does not support mode {mode} (supports {profile['modes']})")
    out = make_run_dir(out_root, run_dir_name(profile["name"], mode, bias, engine.is_mock))
    lock, lock_sha = load_lock()
    meta = {"tool": {"name": "asrbench", "version": __version__}, "started_at": now_jst(),
            "python": sys.version, "platform": platform.platform(), "mock": engine.is_mock,
            "label": MOCK_LABEL if engine.is_mock else "real inference",
            "mode": mode, "comparability": COMPARABILITY[mode], "latency_definition": LATENCY_DEFINITION,
            "bias": bias, "profile": profile, "profile_file": profile_meta,
            "lock": {"sha256": lock_sha, "model": lock["models"].get(profile.get("lock_model")),
                     "python_packages": lock["python_packages"], "sources": lock["sources"]},
            "engine": engine.describe(), "clock": clock.kind,
            "endpoint": profile.get("endpoint"),
            "dataset": {"path": str(dataset_path) if dataset_path else None,
                        "sha256": audio.sha256_file(dataset_path) if dataset_path else None},
            "percentile_method": metrics.PERCENTILE_METHOD, **(extra_meta or {})}
    phrases_on = bias == "on"
    prompt_chars = profile.get("pipeline", {}).get("prompt_previous_chars", 0)
    history, records = {}, []
    with open(out / "events.jsonl", "x", encoding="utf-8") as evf, \
            open(out / "utterances.jsonl", "x", encoding="utf-8") as uf:
        for item in items:
            phrases = tuple(item.nickname_candidates) if phrases_on else ()
            prev = history.get(item.session_id, "")
            prompt = prev[-prompt_chars:] if engine.kind == "faster_whisper" and prompt_chars and prev else None
            base = {"utt_id": item.utt_id, "session_id": item.session_id, "order": item.order,
                    "audio_path": str(item.audio_path), "nickname_candidates": list(item.nickname_candidates),
                    "phrases": list(phrases), "prompt": prompt}
            events, t0 = [], None
            try:
                inp = make_input(item)
                base.update(audio_sha256=inp.audio_sha256, duration_ms=audio.samples_to_ms(len(inp.samples)))
                rec, events, t0 = recognize(engine, profile, mode, item, inp, clock, phrases, prompt)
            except Exception as e:  # recorded per utterance; the run continues
                rec = {"finals": [], "hypothesis": None, "raw_text": None, "failure": f"{type(e).__name__}: {e}"}
                events = getattr(e, "events", [])
            rec = {**base, **rec}
            add_latency(rec, item.truth, events, t0)
            if rec["hypothesis"]:
                history[item.session_id] = prev + rec["hypothesis"]
            rec["replay_origin_t"] = t0
            rec["truth"] = {"reference": item.truth.reference, "reference_verified": item.truth.reference_verified,
                            "speech_start_ms": item.truth.speech_start_ms, "speech_end_ms": item.truth.speech_end_ms,
                            "spoken_nicknames": item.truth.spoken_nicknames, "synthetic": item.synthetic}
            for i, ev in enumerate(events):
                evf.write(_jdump({"utt_id": item.utt_id, "i": i, "t": ev["t"], "t_rel_ms": _ms(ev["t"], t0),
                                  "type": ev["type"], "raw": ev["raw"]}) + "\n")
            uf.write(_jdump(rec) + "\n")
            records.append(rec)
    meta["finished_at"] = now_jst()
    meta["vram"] = vram_summary() if vram_summary else {"label": "not sampled" + (" (mock)" if engine.is_mock else "")}
    per, summary = score(records)
    with open(out / "run.json", "x", encoding="utf-8") as f:
        f.write(json.dumps({"meta": meta, "summary": summary, "per_utterance": per},
                           ensure_ascii=False, indent=2, default=str))
    return out, summary


def rescore(run_dir, items=None):
    """Score existing results; optional dataset items replace the stored truth snapshot."""
    run_dir = Path(run_dir)
    with open(run_dir / "utterances.jsonl", encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    source = "stored truth snapshot"
    if items is not None:
        by_id = {it.utt_id: it for it in items}
        missing = [r["utt_id"] for r in records if r["utt_id"] not in by_id]
        if missing:
            raise ValueError(f"dataset lacks utterances {missing}")
        for r in records:
            t = by_id[r["utt_id"]].truth
            r["truth"] = {**r["truth"], "reference": t.reference, "spoken_nicknames": t.spoken_nicknames}
        source = "dataset"
    per, summary = score(records)
    path = run_dir / f"score-{datetime.now(JST).strftime('%Y%m%dT%H%M%S')}-{os.urandom(2).hex()}.json"
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps({"scored_at": now_jst(), "truth_source": source, "summary": summary,
                            "per_utterance": per}, ensure_ascii=False, indent=2, default=str))
    return path, summary
