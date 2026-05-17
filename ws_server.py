"""
ws_server.py (개선판)

원본 대비 개선점:
  1. 고정 청크 방식 → 무음 감지 + 타임아웃 기반 flush 로 교체
     - 무음 청크 수신 시 즉시 flush (버퍼에 음성이 있을 때만)
     - 일정 시간 오디오 없으면 watchdog 이 강제 flush
  2. Whisper 추론을 ThreadPoolExecutor 에서 실행 → 이벤트 루프 블로킹 방지
  3. 피크 필터링 → 무음 청크는 GPU 에 전혀 보내지 않음
  4. 연결 종료 시 finish() 로 미확정 텍스트 최종 출력
  5. ASRManager.MIN_CHARS 를 환경변수로 조정 가능

프로토콜:
  Client → Server : PCM16LE mono 16kHz raw bytes (binary)
  Server → Client : 인식된 텍스트 (text, UTF-8)

엔드포인트:
  ws://<host>:<port>/asr?session_id=<id>

환경변수 (.env):
  WHISPER_HOST              서버 바인딩 주소 (기본: 0.0.0.0)
  WHISPER_PORT              서버 포트        (기본: 8100)
  WHISPER_LOG_LEVEL         로그 레벨        (기본: INFO)
  WHISPER_SILENCE_FLUSH_SEC 무음 타임아웃(초) (기본: 1.5)
  WHISPER_MAX_BUFFER_SEC    최대 버퍼 길이(초)(기본: 10.0)
  WHISPER_SILENCE_PEAK      무음 판단 피크    (기본: 0.02)
  WHISPER_MIN_SPEECH_PEAK   Whisper 최소 피크 (기본: 0.001)
  WHISPER_MIN_CHARS         최소 출력 글자 수 (기본: 1)
  그 외 기존 WHISPER_* 환경변수 모두 유효
"""

import asyncio
import logging
import os
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

logger.setLevel(LOG_LEVEL)

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
logger.info(f"Whisper 모델 로드 중: model={_args.model} backend={_args.backend} lan={_args.lan}")
_asr_singleton, _ = asr_factory(_args)
logger.info("Whisper 모델 로드 완료")

# GPU 직렬화를 위한 단일 스레드 executor
_executor = ThreadPoolExecutor(max_workers=1)


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def pcm_to_float32(raw_bytes: bytes) -> np.ndarray:
    return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def chunk_is_silence(audio: np.ndarray) -> bool:
    return float(np.max(np.abs(audio))) < SILENCE_PEAK_THRESH


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


# ── WebSocket 핸들러 ──────────────────────────────────────────────────────────
async def asr_handler(websocket, path: str = ""):
    # websockets 최신 버전 호환
    if not path:
        path = getattr(websocket, "path", "") or getattr(
            getattr(websocket, "request", None), "path", ""
        )
    params = parse_qs(urlparse(path).query)
    session_id = (params.get("session_id") or ["unknown"])[0]
    logger.info(f"[{session_id}] 연결됨")

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
        logger.info(f"[{session_id}] 핸들러 종료")


# ── 진입점 ────────────────────────────────────────────────────────────────────
async def main():
    logger.info(f"ws_server 시작: ws://{HOST}:{PORT}/asr")
    # ping_interval=None: 로컬 서비스에서 자동 keepalive ping 비활성화
    # Go 클라이언트(gorilla/websocket)가 ping에 응답하지 않아 1011 오류 발생 방지
    async with websockets.serve(asr_handler, HOST, PORT, ping_interval=None):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
