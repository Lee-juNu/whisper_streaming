"""세그먼트 모드의 일찍 닫기·발화 신호·꼬리 반복 접기 (npc_panopticon/docs/early_endpoint_plan.md P3).

ws_server 는 import 때 ASR 백엔드를 붙이므로 **whisper 컨테이너 안에서** 돈다:
  docker cp ws_server.py tests/test_ws_segment.py <샘플 wav> npc_whisper:/tmp/wt/
  docker exec -w /app -e PYTHONPATH=/tmp/wt:/app -e WS_TEST_WAV=/tmp/wt/sample.wav \\
      npc_whisper python /tmp/wt/test_ws_segment.py

지킬 것: ① 기본값(둘 다 끔)은 예전과 같다 — 신호 없음, 1.5초 확정
② 신호를 켜면 start → end → final 순서이고, 버려진 발화도 end 는 나간다
③ 일찍 닫기를 켜면 그 쉼에서 갈린다 ④ 꼬리 반복은 접되 사람이 하는 반복은 남긴다.
"""
import asyncio
import json
import os
import sys
import threading
import unittest
import wave
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ws_server as W  # noqa: E402

SR = 16000


def speech_clips():
    w = wave.open(os.environ.get("WS_TEST_WAV", "/tmp/wt/sample.wav"))
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if w.getnchannels() > 1:
        a = a[::w.getnchannels()]
    if w.getframerate() != SR:
        a = np.interp(np.arange(0, len(a), w.getframerate() / SR), np.arange(len(a)), a).astype(np.int16)
    return a[:32000], a[32000:64000]


def sil(sec):
    return np.zeros(int(SR * sec), dtype=np.int16)


class FakeWS:
    def __init__(self, stream):
        self.stream, self.sent = stream, []

    def __aiter__(self):
        async def gen():
            for i in range(0, len(self.stream), 1600):
                yield self.stream[i:i + 1600].tobytes()
                await asyncio.sleep(0)
        return gen()

    async def send(self, msg):
        self.sent.append(json.loads(msg))


def run(stream, *, signals, flush):
    W.SPEECH_SIGNALS = signals
    W.SEGMENT_FLUSH_SEC = flush
    W.SEGMENT_FLUSH_REASON = "early" if flush < W.SILENCE_FLUSH_SEC else "silence"
    ws = FakeWS(stream)
    asyncio.run(W._run_segment_mode(ws, "test"))
    return [m["type"] for m in ws.sent], [m for m in ws.sent if m["type"] == "final"]


class Segment(unittest.TestCase):
    def setUp(self):
        self.A, self.B = speech_clips()

    def test_default_is_unchanged(self):
        types, finals = run(np.concatenate([self.A, sil(0.8), self.B, sil(1.6)]), signals=False, flush=1.5)
        self.assertEqual(types, ["final"])              # 0.8초 쉼으로는 안 갈린다, 신호 없음
        self.assertEqual(finals[0]["reason"], "silence")

    def test_signals_wrap_each_final(self):
        types, _ = run(np.concatenate([self.A, sil(1.6), sil(0.5), self.B, sil(1.6)]), signals=True, flush=1.5)
        self.assertEqual(types, ["speech_start", "speech_end", "final"] * 2)

    def test_dropped_noise_still_ends(self):
        click = np.zeros(int(SR * 0.2), dtype=np.int16)
        click[100:200] = 20000                          # 0.4초 미만 — 전사 없이 버려진다
        types, finals = run(np.concatenate([click, sil(1.6)]), signals=True, flush=1.5)
        self.assertEqual(types, ["speech_start", "speech_end"])
        self.assertEqual(finals, [])

    def test_early_flush_splits_at_the_pause(self):
        _, finals = run(np.concatenate([self.A, sil(0.8), self.B, sil(1.6)]), signals=False, flush=0.6)
        self.assertEqual(len(finals), 2)
        self.assertEqual(finals[0]["reason"], "early")


class TailRepeat(unittest.TestCase):
    def test_collapses_nemotron_tail_loops(self):
        self.assertEqual(W._collapse_tail_repeat("約束を守って帰ってきてきてきてきてきて"), "約束を守って帰ってきて")
        self.assertEqual(W._collapse_tail_repeat("ご主人さんさんさんさんさん"), "ご主人さん")
        self.assertEqual(W._collapse_tail_repeat("帰ってきてきてきてきてきて。"), "帰ってきて。")

    def test_keeps_human_repetition(self):
        for text in ("はいはいはいはい", "うんうん、そうだね", "ドキドキしちゃう", "まあまあかな", "あはははは"):
            self.assertEqual(W._collapse_tail_repeat(text), text)


class AudioRanges(unittest.TestCase):
    """합성 PCM과 가짜 전사로 연결별 위치·드롭·동시 수신을 검증한다. WAV 불필요."""

    def setUp(self):
        self.speech = np.full(6400, 4000, dtype=np.int16).tobytes()
        self.settings = patch.multiple(W, SPEECH_SIGNALS=False, SEGMENT_FLUSH_SEC=10.0,
                                       MAX_UTTERANCE_SEC=20.0, MIN_UTTERANCE_SEC=0.4)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def run_messages(self, messages, transcribe=None):
        class ScriptedWS:
            def __init__(self):
                self.sent = []

            def __aiter__(self):
                async def generate():
                    for message in messages:
                        if callable(message):
                            await message()
                        else:
                            yield message
                        await asyncio.sleep(0)
                return generate()

            async def send(self, message):
                self.sent.append(json.loads(message))

        ws = ScriptedWS()
        with patch.object(W, "_transcribe_utterance", side_effect=transcribe or (lambda audio, prompt: "待って")):
            asyncio.run(W._run_segment_mode(ws, "range-test"))
        return [(m["audio_start_sample"], m["audio_end_sample"]) for m in ws.sent if m["type"] == "final"]

    def test_separate_utterances_include_leading_silence(self):
        ranges = self.run_messages([self.speech, '{"type":"eou"}', sil(0.1).tobytes(),
                                    self.speech, '{"type":"eou"}'])
        self.assertEqual(ranges, [(0, 6400), (6400, 14400)])

    def test_dropped_noise_and_silence_still_advance_offset(self):
        ranges = self.run_messages([b"", b"\x00", self.speech[:6400], '{"type":"eou"}',
                                    sil(0.4).tobytes(), '{"type":"eou"}', self.speech])
        self.assertEqual(ranges, [(9600, 16000)])

    def test_filtered_transcript_still_advances_offset(self):
        results = iter(["", "待って"])
        ranges = self.run_messages([self.speech, '{"type":"eou"}', self.speech],
                                   lambda audio, prompt: next(results))
        self.assertEqual(ranges, [(6400, 12800)])

    def test_maximum_split_and_new_connection_reset(self):
        with patch.object(W, "MAX_UTTERANCE_SEC", 0.4):
            self.assertEqual(self.run_messages([self.speech, self.speech]), [(0, 6400), (6400, 12800)])
        self.assertEqual(self.run_messages([self.speech]), [(0, 6400)])

    def test_timeout_snapshot_is_not_extended_by_pcm_during_inference(self):
        started, received_next = threading.Event(), threading.Event()

        def transcribe(audio, prompt):
            if not started.is_set():
                started.set()
                if not received_next.wait(2):
                    raise AssertionError("next PCM was not received during inference")
            return "待って"

        async def await_inference():
            for _ in range(200):
                if started.is_set():
                    return
                await asyncio.sleep(0.01)
            self.fail("timeout did not start inference")

        async def mark_next_received():
            received_next.set()

        with patch.object(W, "SEGMENT_FLUSH_SEC", 0.01):
            ranges = self.run_messages([self.speech, await_inference, self.speech,
                                        mark_next_received, '{"type":"eou"}'], transcribe)
        self.assertEqual(ranges, [(0, 6400), (6400, 12800)])


if __name__ == "__main__":
    unittest.main()
