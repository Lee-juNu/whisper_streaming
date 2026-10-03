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
import unittest
import wave

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


if __name__ == "__main__":
    unittest.main()
