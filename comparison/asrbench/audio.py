"""WAV I/O, hashing, common host energy endpointer, synthetic audio and paced replay."""
import array
import hashlib
import math
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass

from . import SAMPLE_RATE


class WavFormatError(ValueError):
    pass


def ms_to_samples(ms):
    return int(round(ms * SAMPLE_RATE / 1000.0))


def samples_to_ms(n):
    return n * 1000.0 / SAMPLE_RATE


def wav_info(path):
    with wave.open(str(path), "rb") as w:
        info = {"channels": w.getnchannels(), "sample_width": w.getsampwidth(),
                "rate": w.getframerate(), "comptype": w.getcomptype(), "frames": w.getnframes()}
    if (info["channels"], info["sample_width"], info["rate"], info["comptype"]) != (1, 2, SAMPLE_RATE, "NONE"):
        raise WavFormatError(f"{path}: need PCM16 {SAMPLE_RATE}Hz mono uncompressed WAV, got {info}")
    info["duration_ms"] = samples_to_ms(info["frames"])
    return info


def read_wav(path):
    wav_info(path)
    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    a = array.array("h")
    a.frombytes(data)
    if sys.byteorder == "big":
        a.byteswap()
    return a


def write_new_wav(path, samples):
    """Write PCM16 mono WAV; refuses to overwrite (mode 'x')."""
    with open(path, "xb") as f, wave.open(f, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm16le(samples))


def pcm16le(samples):
    if sys.byteorder == "big":
        samples = array.array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def peak(samples):
    if not len(samples):
        return 0.0
    return max(max(samples), -min(samples)) / 32768.0


def silence(ms):
    return array.array("h", bytes(2 * ms_to_samples(ms)))


def tone(ms, freq=220.0, amp=0.3):
    w = 2 * math.pi * freq / SAMPLE_RATE
    a = amp * 32767
    return array.array("h", (int(a * math.sin(w * i)) for i in range(ms_to_samples(ms))))


@dataclass(frozen=True)
class EndpointConfig:
    """Mirror of the production host endpointer parameters.

    frame_ms is the peak-evaluation granularity; the production client chunk size is not
    reproduced (declared assumption, recorded in every run)."""
    peak_threshold: float = 0.02
    trailing_silence_ms: int = 1500
    min_segment_ms: int = 400
    max_segment_ms: int = 20000
    frame_ms: int = 100
    min_audio_peak: float = 0.001  # onset gate: buffering starts at the first frame with peak >= this

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Segment:
    start: int       # buffer start: first frame with peak >= min_audio_peak (may be quiet, < peak_threshold)
    voice_end: int   # end of last voiced frame (== start if the buffer never reached peak_threshold)
    endpoint: int    # sample at which the endpoint is decided (= segment buffer end)
    reason: str      # trailing_silence | max_duration | eof_unterminated

    def as_ms(self):
        return {"start_ms": samples_to_ms(self.start), "voice_end_ms": samples_to_ms(self.voice_end),
                "endpoint_ms": samples_to_ms(self.endpoint), "reason": self.reason}


def find_segments(samples, cfg):
    """Production-like host endpointer. Buffering starts at the first frame whose peak is
    >= min_audio_peak (quiet audio below peak_threshold is included; all-zero audio never opens
    a buffer). Frames >= peak_threshold are voice; the buffer ends after trailing_silence of
    sub-threshold frames (counted from the last voiced frame, or from the buffer start if none),
    or is force-cut at max_segment. Buffers shorter than min_segment (BUFFER length, not voiced
    span) are filtered and returned separately so they never affect grading."""
    frame = ms_to_samples(cfg.frame_ms)
    sil, mn, mx = (ms_to_samples(cfg.trailing_silence_ms), ms_to_samples(cfg.min_segment_ms),
                   ms_to_samples(cfg.max_segment_ms))
    segs, short = [], []
    start = last = None
    n = len(samples)

    def close(end, reason):
        s = Segment(start, last if last is not None else start, end, reason)
        (segs if end - start >= mn or reason == "max_duration" else short).append(s)

    for pos in range(0, n, frame):
        end = min(pos + frame, n)
        pk = peak(samples[pos:end])
        if start is None:
            if pk < cfg.min_audio_peak:
                continue
            start, last = pos, None
        voiced = pk >= cfg.peak_threshold
        if voiced:
            last = end
        if not voiced and end - (last if last is not None else start) >= sil:
            close(end, "trailing_silence")
            start = None
        elif end - start >= mx:
            close(end, "max_duration")
            start = None
    if start is not None:
        close(n, "eof_unterminated")
    return segs, short


def classify_single(segments):
    """Replay-controlled grades exactly one cleanly terminated segment; anything else is rejected."""
    if not segments:
        return None, "no_segment_detected"
    if any(s.reason == "max_duration" for s in segments):
        return None, "exceeds_max_segment"
    if any(s.reason == "eof_unterminated" for s in segments):
        return None, "insufficient_trailing_silence"
    if len(segments) > 1:
        return None, "multiple_segments"
    return segments[0], None


class RealClock:
    kind = "time.monotonic"

    def now(self):
        return time.monotonic()

    def sleep_until(self, t):
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)


class FakeClock:
    """Deterministic clock for tests/mock: sleeping just advances time."""
    kind = "fake"

    def __init__(self, start=1000.0):
        self._t = start
        self._lock = threading.Lock()

    def now(self):
        with self._lock:
            return self._t

    def sleep_until(self, t):
        with self._lock:
            self._t = max(self._t, t)

    def advance(self, dt):
        with self._lock:
            self._t += dt


def paced_replay(samples, packet_samples, clock, on_packet, stop=None):
    """Deliver samples paced by the sample clock: a packet is delivered once its last sample
    'exists' (t0 + end/SR). Returns (t0 replay origin, max send lag ms)."""
    stop = len(samples) if stop is None else min(stop, len(samples))
    t0 = clock.now()
    max_lag = 0.0
    pos = 0
    while pos < stop:
        end = min(pos + packet_samples, stop)
        target = t0 + end / SAMPLE_RATE
        clock.sleep_until(target)
        max_lag = max(max_lag, clock.now() - target)
        on_packet(samples[pos:end], end)
        pos = end
    return t0, max_lag * 1000.0
