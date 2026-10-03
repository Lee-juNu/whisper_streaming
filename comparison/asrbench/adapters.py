"""Engine adapters.

* FasterWhisperEngine  - in-process faster-whisper (optional deps imported lazily here only).
* NemotronEngine       - NeMo-Speech.cpp server: HTTP file transcription + realtime WebSocket.
* WSClient             - minimal RFC 6455 client (stdlib) for the loopback test endpoint.

Engines receive only ``RecognitionInput`` (audio + bias candidates); references, spoken
nickname annotations and manual speech times are never passed in.
"""
import base64
import hashlib
import json
import os
import socket
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import SAMPLE_RATE
from .audio import peak, pcm16le, sha256_bytes, sha256_file
from .guards import find_placeholders, validate_loopback_url

EV_DELTA = "conversation.item.input_audio_transcription.delta"
EV_COMPLETED = "conversation.item.input_audio_transcription.completed"
EV_COMMITTED = "input_audio_buffer.committed"


class EngineUnavailable(RuntimeError):
    """Engine cannot be constructed (missing optional package / local model)."""


class ProtocolError(RuntimeError):
    pass


class PhaseTimeout(ProtocolError):
    pass


@dataclass(frozen=True)
class RecognitionInput:
    """Everything an engine may see. Deliberately has no reference / spoken-nickname /
    manual speech-time fields."""
    utt_id: str
    session_id: str
    audio_path: Path
    audio_sha256: str
    samples: object               # array('h') PCM16 16 kHz mono
    nickname_candidates: tuple

    def wav_bytes(self):
        data = Path(self.audio_path).read_bytes()
        if sha256_bytes(data) != self.audio_sha256:
            raise ProtocolError(f"{self.audio_path} changed during the run")
        return data


# --------------------------------------------------------------------------- faster-whisper

def check_local_model_dir(path):
    if find_placeholders(str(path)):
        raise EngineUnavailable(f"model dir is a placeholder: {path}")
    p = Path(path)
    missing = [n for n in ("model.bin", "config.json") if not (p / n).is_file()]
    if missing:
        raise EngineUnavailable(f"{p}: not a local CTranslate2 model dir (missing {missing}); download it "
                                "manually with the pinned revision (README) - nothing is auto-downloaded")
    return p


class FasterWhisperEngine:
    kind = "faster_whisper"
    is_mock = False

    def __init__(self, profile, postfilter, perf=time.perf_counter):
        self.model_dir = check_local_model_dir(profile["model"]["local_model_dir"])
        try:
            import numpy as np
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise EngineUnavailable("faster-whisper/numpy not importable: install requirements-optional.txt "
                                    "into an isolated venv after approval") from e
        self.np, self.perf = np, perf
        self.opts = dict(profile["transcribe"])
        self.pipe = profile["pipeline"]
        self.postfilter = postfilter
        self.device, self.compute_type = profile["device"], profile["compute_type"]
        t = perf()
        self.model = WhisperModel(str(self.model_dir), device=self.device, compute_type=self.compute_type,
                                  local_files_only=True)
        self.load_ms = (perf() - t) * 1000

    def describe(self):
        return {"engine": self.kind, "model_dir": str(self.model_dir), "load_ms": self.load_ms,
                "device": self.device, "compute_type": self.compute_type,
                "model_bin_sha256": sha256_file(self.model_dir / "model.bin"),
                "config_json_sha256": sha256_file(self.model_dir / "config.json"),
                "transcribe_options": self.opts, "pipeline": self.pipe, "postfilter": self.postfilter.meta}

    def decode_segment(self, inp, samples, prompt, phrases):
        pk = peak(samples)
        if pk < self.pipe["min_audio_peak"]:
            return {"text": "", "raw_text": "", "processing_ms": 0.0,
                    "detail": {"skipped": "below_min_audio_peak", "peak": pk}}
        x = self.np.frombuffer(pcm16le(samples), dtype="<i2").astype(self.np.float32) / 32768.0
        kw = dict(self.opts, initial_prompt=prompt or None, hotwords=" ".join(phrases) if phrases else None)
        t = self.perf()
        it, info = self.model.transcribe(x, **kw)
        segs = list(it)  # generator: decoding happens while consuming it
        ms = (self.perf() - t) * 1000
        keep = [s for s in segs if s.no_speech_prob <= self.pipe["max_no_speech_prob"]
                and s.avg_logprob >= self.pipe["min_avg_logprob"]]
        raw = "".join(s.text for s in keep).strip()
        return {"text": self.postfilter.apply(raw), "raw_text": raw, "processing_ms": ms,
                "detail": {"all_segments_text": "".join(s.text for s in segs),
                           "segments": [{"text": s.text, "start": s.start, "end": s.end,
                                         "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob,
                                         "temperature": getattr(s, "temperature", None), "kept": s in keep}
                                        for s in segs],
                           "language": getattr(info, "language", None)}}

    def transcribe_file(self, inp, prompt, phrases):
        return self.decode_segment(inp, inp.samples, prompt, phrases)


# --------------------------------------------------------------------------- HTTP / WebSocket

class LoopbackHTTP:
    def __init__(self, base_url, timeout):
        self.base = validate_loopback_url(base_url, ("http",)).rstrip("/")
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via proxy

    def _open(self, req):
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raise ProtocolError(f"HTTP {e.code} {req.full_url}: {e.read()[:500]!r}")
        except urllib.error.URLError as e:
            raise ProtocolError(f"cannot reach {req.full_url}: {e.reason}. Start the dedicated test server "
                                "manually (see `server-command`); the harness never starts it")
        try:
            return json.loads(body)
        except ValueError:
            return {"_text": body.decode("utf-8", "replace")}

    def get(self, path):
        return self._open(urllib.request.Request(self.base + path, method="GET"))

    def post_multipart(self, path, fields, file_bytes, filename="audio.wav", content_type="audio/wav"):
        b = "asrbench" + os.urandom(12).hex()
        body = bytearray()
        for k, v in fields.items():
            body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                 f"Content-Type: {content_type}\r\n\r\n").encode() + file_bytes + f"\r\n--{b}--\r\n".encode()
        req = urllib.request.Request(self.base + path, data=bytes(body), method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={b}"})
        return self._open(req)


def _mask(payload, key):
    n = len(payload)
    if not n:
        return b""
    k = (key * (n // 4 + 1))[:n]
    return (int.from_bytes(payload, "little") ^ int.from_bytes(k, "little")).to_bytes(n, "little")


class WSClient:
    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, sock, leftover=b"", mask_outgoing=True):
        self.sock, self.buf, self.mask_outgoing = sock, bytearray(leftover), mask_outgoing
        self.send_lock = threading.Lock()

    @classmethod
    def connect(cls, url, timeout):
        u = urllib.parse.urlsplit(validate_loopback_url(url, ("ws",)))
        sock = socket.create_connection((u.hostname, u.port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
                      f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(4096)
            if not chunk or len(data) > 65536:
                sock.close()
                raise ProtocolError("WebSocket handshake failed: connection closed or oversized header")
            data += chunk
        head, rest = data.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:])}
        accept = base64.b64encode(hashlib.sha1((key + cls.GUID).encode()).digest()).decode()
        if " 101 " not in lines[0] + " " or headers.get("sec-websocket-accept") != accept:
            sock.close()
            raise ProtocolError(f"WebSocket handshake rejected: {lines[0]!r}")
        sock.settimeout(None)
        return cls(sock, rest)

    def _send(self, opcode, payload):
        n = len(payload)
        m = 0x80 if self.mask_outgoing else 0
        h = bytearray([0x80 | opcode])
        if n < 126:
            h.append(m | n)
        elif n < 65536:
            h += bytes([m | 126]) + struct.pack("!H", n)
        else:
            h += bytes([m | 127]) + struct.pack("!Q", n)
        if self.mask_outgoing:
            key = os.urandom(4)
            h += key
            payload = _mask(payload, key)
        with self.send_lock:
            self.sock.sendall(bytes(h) + payload)

    def send_json(self, obj):
        self._send(1, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def send_binary(self, data):
        self._send(2, data)

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("socket closed")
            self.buf += chunk
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def recv(self):
        """Return (opcode, payload) for a complete text(1)/binary(2) message or close(8)."""
        frags, op0 = [], None
        while True:
            b0, b1 = self._read(2)
            fin, op, n = b0 & 0x80, b0 & 0x0F, b1 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read(8))[0]
            key = self._read(4) if b1 & 0x80 else None
            payload = self._read(n)
            if key:
                payload = _mask(payload, key)
            if op == 9:
                self._send(10, payload)
                continue
            if op == 10:
                continue
            if op == 8:
                return 8, payload
            if op != 0:
                op0, frags = op, []
            frags.append(payload)
            if fin:
                return op0, b"".join(frags)

    def close(self):
        try:
            self._send(8, struct.pack("!H", 1000))
        except OSError:
            pass
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class RealtimeSession:
    """One realtime WS session per utterance. Every received message is kept with its
    receive time; nothing returns early on the first completed event."""

    def __init__(self, connect, clock, session_config, timeouts):
        self.connect, self.clock, self.cfg, self.timeouts = connect, clock, session_config, timeouts
        self.events, self.sent = [], []
        self.cond = threading.Condition()
        self.closed = self._closing = False
        self.conn = self.thread = None
        self.commit_index = self.commit_t = None
        self.audio_open = False

    def _record(self, typ, raw, t=None):
        with self.cond:
            self.events.append({"t": self.clock.now() if t is None else t, "type": typ, "raw": raw})
            self.cond.notify_all()

    def _reader(self):
        try:
            while True:
                op, payload = self.conn.recv()
                t = self.clock.now()
                if op == 8:
                    self._record("harness.ws_closed", {"payload_hex": payload.hex()}, t)
                    break
                if op == 1:
                    try:
                        obj = json.loads(payload.decode("utf-8"))
                    except ValueError:
                        self._record("harness.unparsed_text", {"text": payload.decode("utf-8", "replace")}, t)
                        continue
                    typ = obj.get("type") if isinstance(obj, dict) else None
                    self._record(typ or "harness.untyped", obj, t)
                else:
                    self._record("harness.binary_frame", {"bytes": len(payload)}, t)
        except Exception as e:  # persisted, never swallowed silently
            if not self._closing:
                self._record("harness.reader_error", {"error": repr(e)})
        finally:
            with self.cond:
                self.closed = True
                self.cond.notify_all()

    def _wait(self, types, start, timeout, phase):
        deadline = time.monotonic() + timeout
        with self.cond:
            while True:
                for ev in self.events[start:]:
                    if ev["type"] in types:
                        return ev
                if self.closed:
                    raise ProtocolError(f"{phase}: connection closed before {sorted(types)}")
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise PhaseTimeout(f"{phase}: no {sorted(types)} within {timeout}s")
                self.cond.wait(rem)

    def _send_json(self, obj):
        self.sent.append({"t": self.clock.now(), "type": obj.get("type")})
        self.conn.send_json(obj)

    def start(self):
        self.conn = self.connect()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        ev = self._wait({"session.created", "error"}, 0, self.timeouts["session_created"], "session.created")
        if ev["type"] == "error":
            raise ProtocolError(f"server error before session.created: {ev['raw']}")
        with self.cond:
            idx = len(self.events)
        self._send_json({"type": "session.update", "session": self.cfg})  # immutable once audio flows
        ev = self._wait({"session.updated", "error"}, idx, self.timeouts["session_updated"], "session.update")
        if ev["type"] == "error":
            raise ProtocolError(f"server rejected session.update (e.g. speech_contexts unsupported): {ev['raw']}")
        self.audio_open = True

    def send_audio(self, data):
        if not self.audio_open:
            raise ProtocolError("audio before session.updated")
        self.conn.send_binary(data)

    def commit(self):
        with self.cond:
            self.commit_index = len(self.events)
            self.commit_t = self.clock.now()
        self._send_json({"type": "input_audio_buffer.commit"})

    def finish(self, timeout=None):
        """Wait for input_audio_buffer.committed (sent after the final). False on timeout."""
        try:
            self._wait({EV_COMMITTED}, self.commit_index, timeout or self.timeouts["final_after_commit"],
                       "final_after_commit")
            return True
        except ProtocolError as e:
            self._record("harness.phase_error", {"error": str(e)})
            return False

    def close(self):
        self._closing = True
        if self.conn:
            self.conn.close()
        if self.thread:
            self.thread.join(2)


# --------------------------------------------------------------------------- Nemotron

class NemotronEngine:
    kind = "nemotron_server"
    is_mock = False

    def __init__(self, profile, clock, http=None, ws_connect=None, perf=time.perf_counter):
        srv = profile["server"]
        self.profile, self.clock, self.perf = profile, clock, perf
        self.timeouts = profile["timeouts_s"]
        self.http = http or LoopbackHTTP(srv["base_http"], self.timeouts["http"])
        url = validate_loopback_url(srv["realtime_url"], ("ws",))
        self.ws_connect = ws_connect or (lambda: WSClient.connect(url, self.timeouts["connect"]))

    def speech_contexts(self, phrases):
        return [{"phrases": list(phrases), "boost": self.profile["bias"]["boost"]}]

    def server_info(self):
        return {"version": self.http.get("/version"), "models": self.http.get("/v1/models")}

    def check_identity(self, info):
        """The server serves its one loaded model regardless of any request field."""
        text = json.dumps(info["models"], ensure_ascii=False)
        bad = [m for m in self.profile["server"]["rejected_model_markers"] if m in text]
        if bad:
            raise ProtocolError(f"/v1/models reports a rejected model {bad}")
        want = self.profile["server"]["expected_model_markers"]
        if not any(m in text for m in want):
            raise ProtocolError(f"/v1/models does not mention any of {want}: {text[:400]}. Check the server's "
                                "--asr.model.path; update expected_model_markers only after manual verification")

    def describe(self):
        return {"engine": self.kind, "base_http": self.http.base, "session_template": self.profile["session"]}

    def transcribe_file(self, inp, prompt, phrases):
        h = self.profile["http"]
        fields = {"language": h["language"], "response_format": h["response_format"], "verbatim": h["verbatim"]}
        if phrases:
            fields["speech_contexts"] = json.dumps(self.speech_contexts(phrases), ensure_ascii=False)
        data = inp.wav_bytes()
        t = self.perf()
        resp = self.http.post_multipart("/v1/audio/transcriptions", fields, data)
        ms = (self.perf() - t) * 1000
        text = resp.get("text") if isinstance(resp, dict) else None
        if not isinstance(text, str):
            raise ProtocolError(f"response has no 'text': {str(resp)[:300]}")
        return {"text": text, "raw_text": text, "processing_ms": ms,
                "detail": {"response": resp, "request_fields": fields}}

    def open_stream(self, inp, phrases, endpointing_ms):
        cfg = dict(self.profile["session"], sample_rate=SAMPLE_RATE, endpointing_ms=endpointing_ms)
        if phrases:
            cfg["speech_contexts"] = self.speech_contexts(phrases)
        return RealtimeSession(self.ws_connect, self.clock, cfg, self.timeouts)
