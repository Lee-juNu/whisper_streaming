"""
ws_server.py — 마이크 PCM 을 받아 텍스트로 돌려주는 WebSocket ASR 서버.

모드가 둘이고 **세그먼트가 기본**이다(WHISPER_MODE).

  segment (기본)  발화 한 덩어리를 모아 **한 번에** 전사한다. 경계는 클라이언트가
                  이미 VAD 로 알고 있으므로 {"type":"eou"} 로 받고, 안 보내는
                  클라이언트를 위해 서버가 무음 폴백을 갖는다. 발화당 추론 1회.
  stream          기존 LocalAgreement 스트리밍. 롤백용으로 남겨 둔다.

세그먼트로 옮긴 이유는 스트리밍의 구조적 한계다 — process_iter() 는 직전 추론과
단어가 일치할 때만 확정하므로 발화의 **마지막 단어가 다음 오디오를 기다린다**.
클라이언트는 발화가 끝나면 아무것도 안 보내므로 그 오디오가 영영 안 오고,
사람이 「흠흠」 하고 소리를 내야 문장이 완성됐다. 게다가 whisper 는 입력을 30초로
패딩해 인코딩하므로 같은 오디오를 3초마다 다시 미는 비용이 발화당 1회와 같다.

프로토콜:
  Client → Server : PCM16LE mono 16kHz raw bytes (binary)
                    제어 프레임 (text, JSON) — 세그먼트 모드 전용, 없어도 동작한다
                      {"type":"eou"}           발화 끝. 지금까지 받은 것을 확정한다
                      {"type":"speech_start"}  새 발화 시작. 남은 앞 발화를 먼저 닫는다
  Server → Client : segment 모드 {"type":"final","text":...,"duration":...,"reason":...}
                    stream  모드 인식된 텍스트 (text, UTF-8)  ← 기존과 동일

엔드포인트:
  ws://<host>:<port>/asr?session_id=<id>

환경변수 (.env):
  WHISPER_HOST              서버 바인딩 주소   (기본: 0.0.0.0)
  WHISPER_PORT              서버 포트          (기본: 8100)
  WHISPER_LOG_LEVEL         로그 레벨          (기본: INFO)
  WHISPER_BACKEND           faster-whisper | nemotron (기본: faster-whisper)
                            nemotron 은 세그먼트 모드 전용 — 발화 WAV 를 nemo-speech 서버로
                            보낸다(asr_backends.NemotronASR, NEMO_URL / NEMO_LANGUAGE)
  WHISPER_MODE              segment | stream   (기본: segment)
  WHISPER_SILENCE_FLUSH_SEC 무음 타임아웃(초)   (기본: 1.5) — 두 모드 공용
  WHISPER_SILENCE_PEAK      무음 판단 피크      (기본: 0.02)
  WHISPER_MIN_SPEECH_PEAK   Whisper 최소 피크   (기본: 0.001)

  [segment 전용]
  WHISPER_MAX_UTTERANCE_SEC 발화 상한(초)       (기본: 20.0) — 넘으면 강제로 끊는다
  WHISPER_MIN_UTTERANCE_SEC 발화 최소 길이(초)  (기본: 0.4)  — 짧으면 버린다(환각 방지)
  WHISPER_MAX_NO_SPEECH     no_speech_prob 컷   (기본: 0.6)
  WHISPER_MIN_AVG_LOGPROB   avg_logprob 컷      (기본: -1.0)
  WHISPER_BLOCKLIST         환각 상투구(쉼표 구분). 비우면 기본 목록
  WHISPER_BEAM_SIZE         beam size           (기본: 5, asr_backends)

  [stream 전용]
  WHISPER_MAX_BUFFER_SEC    최대 버퍼 길이(초)  (기본: 10.0)
  WHISPER_PERIODIC_FLUSH_SEC 주기적 flush(초)   (기본: 3.0)
  WHISPER_MIN_FLUSH_SEC     무음 flush 최소(초) (기본: 0.8)
  WHISPER_MIN_CHARS         최소 출력 글자 수   (기본: 1)

  그 외 기존 WHISPER_* 환경변수 모두 유효
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs

import numpy as np
import websockets
from dotenv import load_dotenv

from manager import ASRManager
from audio import SAMPLING_RATE
from online_processor import OnlineASRProcessor
from vad import VACOnlineASRProcessor
from whisper_online_server import asr_factory, add_shared_args, override_args_with_env

load_dotenv()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ws_server")

# ── 설정 ─────────────────────────────────────────────────────────────────────
HOST                = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT                = int(os.getenv("WHISPER_PORT", "8100"))
LOG_LEVEL           = os.getenv("WHISPER_LOG_LEVEL", "INFO")
SILENCE_FLUSH_SEC   = float(os.getenv("WHISPER_SILENCE_FLUSH_SEC", "1.5"))
MAX_BUFFER_SEC      = float(os.getenv("WHISPER_MAX_BUFFER_SEC", "10.0"))
SILENCE_PEAK_THRESH = float(os.getenv("WHISPER_SILENCE_PEAK", "0.02"))
MIN_SPEECH_PEAK     = float(os.getenv("WHISPER_MIN_SPEECH_PEAK", "0.001"))
MIN_CHARS           = int(os.getenv("WHISPER_MIN_CHARS", "1"))
# 연속 발화 시에도 주기적으로 Whisper 에 밀어 넣는 간격(초)
# 무음이 없어도 이 주기마다 process_iter() 실행 → 중간 결과 출력
PERIODIC_FLUSH_SEC  = float(os.getenv("WHISPER_PERIODIC_FLUSH_SEC", "3.0"))
# 무음 감지로 즉시 flush 할 때 필요한 최소 버퍼 길이(초)
# 이보다 짧으면 Whisper 가 결과를 내지 않고 상태만 꼬이므로 watchdog 에 맡김
MIN_FLUSH_SEC       = float(os.getenv("WHISPER_MIN_FLUSH_SEC", "0.8"))

# ── 세그먼트 모드 설정 ───────────────────────────────────────────────────────
MODE                = os.getenv("WHISPER_MODE", "segment").strip().lower()
# 무음 없이 계속 말할 때의 상한. 넘으면 끊어서 내보낸다 — 안 그러면 아무것도 안 나온다.
MAX_UTTERANCE_SEC   = float(os.getenv("WHISPER_MAX_UTTERANCE_SEC", "20.0"))
# 이보다 짧은 덩어리는 전사하지 않는다. 짧은 잡음 하나에 whisper 가 문장을 지어낸다.
MIN_UTTERANCE_SEC   = float(os.getenv("WHISPER_MIN_UTTERANCE_SEC", "0.4"))
MAX_NO_SPEECH       = float(os.getenv("WHISPER_MAX_NO_SPEECH", "0.6"))
MIN_AVG_LOGPROB     = float(os.getenv("WHISPER_MIN_AVG_LOGPROB", "-1.0"))
# 다음 발화에 물려줄 문맥 길이. 길게 주면 whisper 가 앞 발화를 되풀이한다.
PROMPT_MAX_CHARS    = int(os.getenv("WHISPER_PROMPT_MAX_CHARS", "120"))

# 학습 데이터(자막)에서 온 상투구. 무음·잡음 구간에서 이것들이 통째로 나온다.
# **발화 전체가 이것과 같을 때만** 버린다 — 실제로 말할 수 있는 문장은 넣지 않는다
# (예: 「ありがとうございました」 는 사람이 진짜 하는 말이라 목록에 없다).
_DEFAULT_BLOCKLIST = [
    "ご視聴ありがとうございました", "ご視聴ありがとうございます",
    "ご覧いただきありがとうございます", "最後までご視聴いただきありがとうございました",
    "チャンネル登録をお願いします", "チャンネル登録よろしくお願いします",
    "高評価とチャンネル登録をお願いします", "次回の動画でお会いしましょう",
    "字幕視聴ありがとうございました", "おつかれさまでした",
    "시청해주셔서 감사합니다", "구독과 좋아요 부탁드립니다",
    "Thanks for watching!", "Thank you for watching.",
    "Please subscribe to my channel.",
]
_blocklist_env = os.getenv("WHISPER_BLOCKLIST", "").strip()
BLOCKLIST = (
    [x.strip() for x in _blocklist_env.split(",") if x.strip()]
    if _blocklist_env else _DEFAULT_BLOCKLIST
)

logger.setLevel(LOG_LEVEL)
# 백엔드 모듈의 로그(nemo-speech 대기·모델 ID 등)도 같은 레벨로 보이게 한다 — 루트는 WARNING 이라 묻힌다.
logging.getLogger("asr_backends").setLevel(LOG_LEVEL)

# ASRManager 의 MIN_CHARS 를 환경변수로 덮어씀 (실시간 대화에서 짧은 텍스트도 출력)
ASRManager.MIN_CHARS = MIN_CHARS

# ── ASR 모델 로드 (서버 시작 시 1회) ─────────────────────────────────────────
def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_shared_args(parser)
    args, _ = parser.parse_known_args()
    args = override_args_with_env(args)
    return args

_args = _build_args()
logger.info(f"ASR 로드 중: model={_args.model} backend={_args.backend} lan={_args.lan}")
_asr_singleton, _ = asr_factory(_args)
logger.info(f"ASR 모델 로드 완료 (backend={_args.backend})")

# GPU 직렬화를 위한 단일 스레드 executor
_executor = ThreadPoolExecutor(max_workers=1)


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def pcm_to_float32(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def chunk_is_silence(audio: np.ndarray) -> bool:
    return float(np.max(np.abs(audio))) < SILENCE_PEAK_THRESH


# 발화 전체가 상투구와 같은지 볼 때 쓰는 정규화(공백·구두점 제거).
_NORMALIZE_RE = re.compile(r"[\s。、．，,.!?！？…♪「」『』\-~ー]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text).lower()


def _is_looping(text: str) -> bool:
    """같은 조각이 되풀이되는 whisper 특유의 루프인지.

    무음을 물리면 「あああああ」나 짧은 구를 끝없이 되풀이한다. 스트리밍에서는
    「두 번 연속 같은 단어」 조건이 이것을 우연히 걸러 줬는데, 발화 단위로는
    한 번의 추론이 그대로 나가므로 여기서 직접 본다.

    반복 횟수 기준을 조각 길이에 따라 다르게 잡는다. 1~2글자짜리를 네댓 번
    거듭하는 것은 사람이 실제로 하는 말이고(「はいはいはいはい」), 3글자 이상의
    구를 네 번 반복하는 것은 사람이 잘 안 한다.
    """
    t = _normalize(text)
    if len(t) < 8:
        return False
    for unit in range(1, 11):
        min_repeat = 6 if unit <= 2 else 4
        if len(t) < unit * min_repeat:
            break
        repeat = len(t) // unit
        if repeat < min_repeat:
            continue
        head = t[:unit]
        if head * repeat == t[: unit * repeat]:
            return True
    return False


def _transcribe_utterance(audio: np.ndarray, init_prompt: str) -> str:
    """발화 한 덩어리를 전사하고 환각을 걸러 텍스트 하나로 돌려준다.

    executor 스레드에서 돈다(GPU 직렬화). 걸러낸 이유는 DEBUG 로그로 남긴다 —
    필터가 실제 발화를 먹고 있는지 나중에 확인할 수 있어야 한다.
    """
    res  = _asr_singleton.transcribe_utterance(audio, init_prompt=init_prompt)
    segs = _asr_singleton.utterance_segments(res)

    kept = []
    for text, no_speech, avg_logprob in segs:
        text = text.strip()
        if not text:
            continue
        if no_speech > MAX_NO_SPEECH:
            logger.debug(f"세그먼트 버림 (no_speech={no_speech:.2f}): {text!r}")
            continue
        if avg_logprob < MIN_AVG_LOGPROB:
            logger.debug(f"세그먼트 버림 (avg_logprob={avg_logprob:.2f}): {text!r}")
            continue
        kept.append(text)

    if not kept:
        return ""

    full = _asr_singleton.sep.join(kept).strip()
    if not full:
        return ""

    norm = _normalize(full)
    if any(norm == _normalize(b) for b in BLOCKLIST):
        logger.debug(f"발화 버림 (상투구): {full!r}")
        return ""
    if _is_looping(full):
        logger.debug(f"발화 버림 (루프): {full!r}")
        return ""

    return full


def _make_online_processor():
    trimming = (_args.buffer_trimming, _args.buffer_trimming_sec)
    if _args.vac:
        return VACOnlineASRProcessor(
            _args.min_chunk_size,
            _asr_singleton,
            None,
            buffer_trimming=trimming,
            logfile=sys.stderr,
        )
    return OnlineASRProcessor(
        _asr_singleton,
        None,
        buffer_trimming=trimming,
        logfile=sys.stderr,
    )


# ── 스트림 모드 핸들러 (WHISPER_MODE=stream) ─────────────────────────────────
async def _run_stream_mode(websocket, session_id: str):
    """LocalAgreement 스트리밍. 세그먼트 모드가 기본이 된 뒤의 **롤백 경로**다.

    구조상의 한계가 하나 있다: process_iter() 는 직전 추론과 단어가 일치할 때만
    확정하므로(online_processor.HypothesisBuffer.flush), 발화의 **마지막 단어는
    다음 오디오가 와야** 확정된다. 무음 타임아웃이 와도 여기서 부르는 것은
    process_iter() 라 꼬리가 남고, finish() 는 연결 종료 때만 불린다.
    그래서 사람이 「흠흠」처럼 소리를 한 번 더 내야 문장이 완성되는 일이 생긴다.
    세그먼트 모드는 이 문제 자체가 없다(발화가 끝나면 그 덩어리를 한 번에 전사한다).
    """
    online  = _make_online_processor()
    online.init()
    manager = ASRManager()

    audio_buffer: list[np.ndarray] = []
    buffer_sec   = 0.0
    last_recv_at = time.monotonic()
    processing   = False  # flush 중 중복 방지

    # ── flush 핵심 로직 ──────────────────────────────────────────────────────
    async def flush(is_final: bool = False):
        nonlocal audio_buffer, buffer_sec, processing

        if processing or not audio_buffer:
            return

        # 버퍼 전체 peak 확인 → 음성 없으면 skip
        conc = np.concatenate(audio_buffer)
        peak = float(np.max(np.abs(conc)))

        audio_buffer = []
        buffer_sec   = 0.0

        if peak < MIN_SPEECH_PEAK:
            logger.debug(f"[{session_id}] flush skip (peak={peak:.4f} too low)")
            return

        processing = True
        try:
            loop = asyncio.get_running_loop()

            if is_final:
                # 남은 오디오를 넣고 finish() → 미확정 텍스트까지 모두 출력
                def _do_final():
                    online.insert_audio_chunk(conc)
                    return online.finish()
                seg = await loop.run_in_executor(_executor, _do_final)
            else:
                # 일반 flush → process_iter() 로 확정된 텍스트만 출력
                def _do_iter():
                    online.insert_audio_chunk(conc)
                    return online.process_iter()
                seg = await loop.run_in_executor(_executor, _do_iter)

            result = manager.handle(seg, is_final)
            if result:
                logger.info(f"[{session_id}] STT: {result!r}")
                try:
                    await websocket.send(result)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"[{session_id}] flush 오류: {e}", exc_info=True)
        finally:
            processing = False

    # ── 감시 태스크: 무음 타임아웃 + 주기적 flush ───────────────────────────
    # 체크 간격을 0.3 s 로 유지하면서 두 조건을 함께 처리
    #   1) SILENCE_FLUSH_SEC 동안 새 프레임이 없으면 → flush (마이크 침묵)
    #   2) PERIODIC_FLUSH_SEC 마다 버퍼에 음성이 쌓여 있으면 → flush (연속 발화 중간 결과)
    async def silence_watchdog():
        last_periodic = time.monotonic()
        while True:
            await asyncio.sleep(0.3)
            now = time.monotonic()

            # 조건 1: 새 프레임 없이 SILENCE_FLUSH_SEC 초 경과
            if audio_buffer and (now - last_recv_at) > SILENCE_FLUSH_SEC:
                logger.debug(f"[{session_id}] 무음 타임아웃 flush ({SILENCE_FLUSH_SEC}s)")
                await flush(is_final=False)
                last_periodic = now
                continue

            # 조건 2: 주기적 flush — 연속 발화로 무음이 없어도 중간 결과 추출
            if audio_buffer and (now - last_periodic) >= PERIODIC_FLUSH_SEC:
                logger.debug(f"[{session_id}] 주기적 flush ({PERIODIC_FLUSH_SEC}s)")
                await flush(is_final=False)
                last_periodic = now

    watchdog = asyncio.create_task(silence_watchdog())

    # ── 메인 수신 루프 ────────────────────────────────────────────────────────
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            try:
                audio = pcm_to_float32(message)
            except Exception as e:
                logger.warning(f"[{session_id}] PCM 변환 오류: {e}")
                continue

            last_recv_at = time.monotonic()
            audio_buffer.append(audio)
            buffer_sec += len(audio) / SAMPLING_RATE

            silence = chunk_is_silence(audio)

            if silence:
                # 무음 청크 → 최소 버퍼(MIN_FLUSH_SEC) 이상 쌓였을 때만 flush
                # 단어 사이 짧은 휴지로 즉시 flush 되면 작은 버퍼가 Whisper 상태를 꼬이게 함
                if buffer_sec >= MIN_FLUSH_SEC and not processing:
                    await flush(is_final=False)
            elif buffer_sec >= MAX_BUFFER_SEC:
                # 최대 버퍼 초과 → 강제 flush
                await flush(is_final=False)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"[{session_id}] 정상 종료")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"[{session_id}] 비정상 종료: {e}")
    except Exception as e:
        logger.error(f"[{session_id}] 오류: {e}", exc_info=True)
    finally:
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass

        # 연결 종료 시 남은 버퍼 최종 flush
        await flush(is_final=True)
        logger.info(f"[{session_id}] 스트림 핸들러 종료")


# ── 세그먼트 모드 핸들러 (WHISPER_MODE=segment, 기본) ────────────────────────
async def _run_segment_mode(websocket, session_id: str):
    """발화 한 덩어리를 모아 **한 번에** 전사한다.

    경계를 추측하지 않는다. 클라이언트가 이미 VAD 로 판정하고 있으므로
    ``{"type":"eou"}`` 한 줄이면 끝이고, 그것을 안 보내는 클라이언트를 위해
    서버가 무음(수신 공백 + 수신한 무음 청크)으로 같은 판정을 폴백으로 갖는다.

    확정 계기는 넷이다 — eou(클라이언트가 알린 발화 끝) / timeout(수신 공백) /
    silence(받은 청크가 계속 무음) / max(상한 초과 강제 컷).
    """
    utterance: list[np.ndarray] = []
    utt_sec         = 0.0
    trailing_sil    = 0.0          # 버퍼 꼬리에 붙어 있는 무음 길이(초)
    last_recv_at    = time.monotonic()
    prev_text       = ""           # 다음 발화의 init_prompt (고유명사 유지용)
    lock            = asyncio.Lock()

    async def finalize(reason: str):
        """버퍼를 비우고 전사해 확정 텍스트를 보낸다. 계기와 무관하게 이 한 곳만 쓴다."""
        nonlocal utterance, utt_sec, trailing_sil, prev_text

        async with lock:
            # 여기에는 await 가 없다 — 수신 루프가 끼어들어 옛 리스트에 덧붙일 틈이 없다.
            if not utterance:
                return
            audio        = np.concatenate(utterance)
            utterance    = []
            utt_sec      = 0.0
            trailing_sil = 0.0

            dur  = len(audio) / SAMPLING_RATE
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0

            # 짧은 잡음 한 조각을 whisper 에 넣으면 문장을 통째로 지어낸다.
            # 스트리밍에서는 「두 번 연속 같은 단어」 조건이 이것을 우연히 걸러 줬다.
            if dur < MIN_UTTERANCE_SEC:
                logger.debug(f"[{session_id}] drop: too short ({dur:.2f}s < {MIN_UTTERANCE_SEC}s, {reason})")
                return
            if peak < MIN_SPEECH_PEAK:
                logger.debug(f"[{session_id}] drop: silent (peak={peak:.4f}, {reason})")
                return

            loop = asyncio.get_running_loop()
            t0   = time.monotonic()
            try:
                text = await loop.run_in_executor(
                    _executor, _transcribe_utterance, audio, prev_text
                )
            except Exception as e:
                logger.error(f"[{session_id}] 전사 오류: {e}", exc_info=True)
                return

            if not text:
                return

            # 다음 발화의 문맥. 통째로 물려주면 whisper 가 앞 발화를 되풀이하므로 꼬리만 남긴다.
            prev_text = text[-PROMPT_MAX_CHARS:]

            infer = time.monotonic() - t0
            logger.info(
                f"[{session_id}] STT({reason}): {text!r} "
                f"[발화 {dur:.1f}s / 추론 {infer:.2f}s]"
            )
            try:
                await websocket.send(json.dumps(
                    {"type": "final", "text": text, "duration": round(dur, 2), "reason": reason},
                    ensure_ascii=False,
                ))
            except Exception:
                pass

    async def handle_control(raw: str):
        """클라이언트가 보내는 제어 프레임. 모르는 것은 조용히 무시한다."""
        try:
            msg = json.loads(raw)
            kind = msg.get("type")
        except Exception:
            logger.debug(f"[{session_id}] 제어 프레임 파싱 실패: {raw[:80]!r}")
            return

        if kind == "eou":
            await finalize("eou")
        elif kind == "speech_start":
            # 새 발화가 시작됐는데 앞 발화가 아직 버퍼에 남아 있으면 먼저 닫는다
            # (EOU 가 유실됐거나 에코 가드로 중간에 끊긴 경우).
            if utterance:
                await finalize("speech_start")
        else:
            logger.debug(f"[{session_id}] 알 수 없는 제어 프레임: {kind!r}")

    async def silence_watchdog():
        """클라이언트가 EOU 를 안 보낼 때의 폴백. 수신이 끊기면 발화 끝으로 본다."""
        while True:
            await asyncio.sleep(0.2)
            if utterance and (time.monotonic() - last_recv_at) > SILENCE_FLUSH_SEC:
                await finalize("timeout")

    watchdog = asyncio.create_task(silence_watchdog())

    try:
        async for message in websocket:
            if isinstance(message, str):
                await handle_control(message)
                continue
            if not isinstance(message, bytes):
                continue

            try:
                audio = pcm_to_float32(message)
            except Exception as e:
                logger.warning(f"[{session_id}] PCM 변환 오류: {e}")
                continue

            if audio.size == 0:
                # 빈 바이너리 프레임. 아래 무음 판정이 빈 배열에서 터진다.
                continue

            chunk_sec    = len(audio) / SAMPLING_RATE
            last_recv_at = time.monotonic()
            utterance.append(audio)
            utt_sec += chunk_sec

            # 무음을 계속 흘려보내는 클라이언트(VAD 가 없는 쪽)도 같은 판정을 받게 한다.
            if chunk_is_silence(audio):
                trailing_sil += chunk_sec
            else:
                trailing_sil = 0.0

            if trailing_sil >= SILENCE_FLUSH_SEC:
                await finalize("silence")
            elif utt_sec >= MAX_UTTERANCE_SEC:
                # 무음 없이 계속 말하는 중. 여기서 끊지 않으면 아무것도 안 나온다.
                await finalize("max")

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"[{session_id}] 정상 종료")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"[{session_id}] 비정상 종료: {e}")
    except Exception as e:
        logger.error(f"[{session_id}] 오류: {e}", exc_info=True)
    finally:
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass

        # 남은 발화는 버리지 않는다 — 소켓이 끊겨 send 가 실패해도 로그에는 남는다.
        await finalize("close")
        logger.info(f"[{session_id}] 세그먼트 핸들러 종료")


# ── WebSocket 진입 핸들러 ────────────────────────────────────────────────────
async def asr_handler(websocket, path: str = ""):
    # websockets 최신 버전 호환
    if not path:
        path = getattr(websocket, "path", "") or getattr(
            getattr(websocket, "request", None), "path", ""
        )
    params = parse_qs(urlparse(path).query)
    session_id = (params.get("session_id") or ["unknown"])[0]
    logger.info(f"[{session_id}] 연결됨 (mode={MODE})")

    if MODE == "stream":
        await _run_stream_mode(websocket, session_id)
    else:
        await _run_segment_mode(websocket, session_id)


# ── 진입점 ────────────────────────────────────────────────────────────────────
async def main():
    logger.info(f"ws_server 시작: ws://{HOST}:{PORT}/asr")
    # ping_interval=None: 로컬 서비스에서 자동 keepalive ping 비활성화
    # Go 클라이언트(gorilla/websocket)가 ping에 응답하지 않아 1011 오류 발생 방지
    async with websockets.serve(asr_handler, HOST, PORT, ping_interval=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
