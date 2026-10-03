"""Synthetic data and mock engines. Everything here is MOCK: tones instead of speech and
scripted outputs. It exercises the pipeline and says nothing about model quality or speed."""
import array
import json
from pathlib import Path

from . import SAMPLE_RATE
from .adapters import EV_COMMITTED, EV_COMPLETED, EV_DELTA
from .audio import peak, silence, tone, write_new_wav

# pattern: (kind, ms). reference strings are invented placeholders, NOT transcripts of the tones.
SYNTHETIC = [
    {"utt_id": "syn-s1-001", "session_id": "syn-s1", "order": 1,
     "pattern": [("sil", 500), ("tone", 1200), ("sil", 2000)],
     "reference": "みーちゃんおはよう", "candidates": ["みーちゃん", "みー", "タロウ"], "spoken": {"みーちゃん": 1}},
    {"utt_id": "syn-s1-002", "session_id": "syn-s1", "order": 2,
     "pattern": [("sil", 400), ("tone", 800), ("sil", 300), ("tone", 700), ("sil", 2000)],
     "reference": "タロウくんとみーちゃん", "candidates": ["みーちゃん", "タロウ", "タロ"],
     "spoken": {"タロウ": 1, "みーちゃん": 1}},
    {"utt_id": "syn-s1-003", "session_id": "syn-s1", "order": 3,
     "pattern": [("sil", 3000)], "reference": None, "candidates": ["みーちゃん"], "spoken": {}},
    {"utt_id": "syn-s2-001", "session_id": "syn-s2", "order": 1,
     "pattern": [("sil", 300), ("tone", 600), ("sil", 2000), ("tone", 600), ("sil", 2000)],
     "reference": "ふたつのはつわ", "candidates": ["ポチ"], "spoken": {}},
    {"utt_id": "syn-s2-002", "session_id": "syn-s2", "order": 2,
     "pattern": [("sil", 200), ("tone", 20500), ("sil", 2000)],
     "reference": "ながいはつわ", "candidates": [], "spoken": None},
    {"utt_id": "syn-s2-003", "session_id": "syn-s2", "order": 3,
     "pattern": [("sil", 300), ("tone", 900), ("sil", 600)],
     "reference": "", "candidates": ["ポチ"], "spoken": {}},
]

# Scripted mock outputs keyed by utt_id (independent of the references above).
MOCK_TEXT = {
    "syn-s1-001": ["みーちゃんおはよ"],
    "syn-s1-002": ["たろうくんとみーちゃんみーちゃん"],
    "syn-s1-003": [""],
    "syn-s2-001": ["ふたつの", "はつわポチ"],
    "syn-s2-002": ["ながいはつわ"],
    "syn-s2-003": ["えっと"],
}


def write_synthetic_dataset(directory):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=False)
    lines = []
    for s in SYNTHETIC:
        samples = array.array("h")
        t = 0
        start = end = None
        for kind, ms in s["pattern"]:
            if kind == "tone":
                start = t if start is None else start
                end = t + ms
                samples += tone(ms)
            else:
                samples += silence(ms)
            t += ms
        write_new_wav(d / f"{s['utt_id']}.wav", samples)
        lines.append({"utt_id": s["utt_id"], "session_id": s["session_id"], "order": s["order"],
                      "audio_path": f"{s['utt_id']}.wav", "reference": s["reference"],
                      "reference_verified": False, "synthetic": True,
                      "reference_kind": "synthetic_placeholder_not_a_transcript",
                      "speech_start_ms": start if start is not None else 0,
                      "speech_end_ms": end if end is not None else t,
                      "nickname_candidates": s["candidates"], "spoken_nicknames": s["spoken"],
                      "notes": "synthetic tone/silence; " + ("silence only" if start is None else "tone bursts")})
    path = d / "dataset.synthetic.jsonl"
    with open(path, "x", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return path


class MockWhisperEngine:
    kind = "faster_whisper"
    is_mock = True

    def __init__(self, clock, processing_ms=150.0, script=None):
        self.clock, self.processing_ms = clock, processing_ms
        self.script = script or MOCK_TEXT
        self.calls = []

    def describe(self):
        return {"engine": "mock_faster_whisper", "mock": True, "processing_ms": self.processing_ms}

    def decode_segment(self, inp, samples, prompt, phrases):
        self.calls.append({"utt_id": inp.utt_id, "prompt": prompt, "phrases": list(phrases),
                           "n_samples": len(samples)})
        self.clock.advance(self.processing_ms / 1000.0)
        text = "".join(self.script.get(inp.utt_id, [""]))
        return {"text": text, "raw_text": text, "processing_ms": self.processing_ms, "detail": {"mock": True}}

    def transcribe_file(self, inp, prompt, phrases):
        return self.decode_segment(inp, inp.samples, prompt, phrases)


class MockStreamSession:
    """Emulates realtime events. With server_endpointing it finalizes after endpointing_ms of
    sub-threshold audio; commit flushes the remainder (possibly an empty final)."""

    def __init__(self, clock, parts, cfg, server_endpointing, latency_s=0.12):
        self.clock, self.parts, self.cfg = clock, list(parts), cfg
        self.server_endpointing, self.latency_s = server_endpointing, latency_s
        self.events, self.sent = [], []
        self.commit_index = self.commit_t = None
        self.voiced_ms = self.silence_ms = 0.0
        self.pending = self.partial_sent = False
        self.part = 0

    def _emit(self, typ, dt=0.0, **raw):
        self.events.append({"t": self.clock.now() + dt, "type": typ, "raw": {"type": typ, **raw}})

    def start(self):
        self._emit("session.created")
        self.sent.append({"t": self.clock.now(), "type": "session.update"})
        self._emit("session.updated", session=self.cfg)

    def _next_part(self):
        text = self.parts[self.part] if self.part < len(self.parts) else ""
        self.part += 1
        return text

    def send_audio(self, data):
        a = array.array("h")
        a.frombytes(data)
        ms = len(a) * 1000.0 / SAMPLE_RATE
        if peak(a) >= 0.02:
            self.voiced_ms += ms
            self.silence_ms = 0.0
            self.pending = True
            if not self.partial_sent and self.voiced_ms >= 300:
                self._emit(EV_DELTA, self.latency_s / 2, delta=(self.parts[0] if self.parts else "")[:2])
                self.partial_sent = True
        else:
            self.silence_ms += ms
            if self.server_endpointing and self.pending and self.silence_ms >= self.cfg["endpointing_ms"]:
                self._emit(EV_COMPLETED, self.latency_s, transcript=self._next_part())
                self.pending = False

    def commit(self):
        self.commit_index = len(self.events)
        self.commit_t = self.clock.now()
        self.sent.append({"t": self.commit_t, "type": "input_audio_buffer.commit"})
        if self.server_endpointing:
            rest = "".join(self.parts[self.part:]) if self.pending else ""
            self.part = len(self.parts)
            self._emit(EV_COMPLETED, self.latency_s, transcript=rest)
        else:
            self._emit(EV_COMPLETED, self.latency_s, transcript="".join(self.parts))
        self._emit(EV_COMMITTED, self.latency_s + 0.001)

    def finish(self, timeout=None):
        return True

    def close(self):
        pass


class MockNemotronEngine:
    kind = "nemotron_server"
    is_mock = True

    def __init__(self, clock, server_endpointing, processing_ms=90.0, script=None):
        self.clock, self.server_endpointing, self.processing_ms = clock, server_endpointing, processing_ms
        self.script = script or MOCK_TEXT
        self.sessions = []

    def describe(self):
        return {"engine": "mock_nemotron_server", "mock": True, "server_endpointing": self.server_endpointing}

    def transcribe_file(self, inp, prompt, phrases):
        self.clock.advance(self.processing_ms / 1000.0)
        text = "".join(self.script.get(inp.utt_id, [""]))
        return {"text": text, "raw_text": text, "processing_ms": self.processing_ms,
                "detail": {"mock": True, "speech_contexts": list(phrases)}}

    def open_stream(self, inp, phrases, endpointing_ms):
        cfg = {"sample_rate": SAMPLE_RATE, "endpointing_ms": endpointing_ms}
        if phrases:
            cfg["speech_contexts"] = [{"phrases": list(phrases), "boost": 3.0}]
        s = MockStreamSession(self.clock, self.script.get(inp.utt_id, [""]), cfg, self.server_endpointing)
        self.sessions.append(s)
        return s
