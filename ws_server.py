"""
ws_server.py

기존 whisper_online_server.py 의 ASR 로직을 WebSocket 으로 래핑.
npc_gateway 의 VoiceStreamHandler 가 이 서버에 연결합니다.

프로토콜:
  Client → Server : PCM16LE mono 16kHz raw bytes (binary)
  Server → Client : 인식된 텍스트 (text, UTF-8)

엔드포인트:
  ws://<host>:<port>/asr?session_id=<id>
"""

import asyncio
import io
import logging
import os
import sys
from urllib.parse import urlparse, parse_qs

import numpy as np
import soundfile
import librosa
import websockets
from dotenv import load_dotenv

from manager import ASRManager
from audio import SAMPLING_RATE
from asr_backends import FasterWhisperASR, WhisperTimestampedASR, OpenaiApiASR
from online_processor import OnlineASRProcessor
from vad import VACOnlineASRProcessor
from whisper_online_server import asr_factory, add_shared_args, override_args_with_env

load_dotenv()

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ws_server")

# ── 설정 (환경변수 / .env) ─────────────────────────────────────────────────────
HOST = os.getenv("WHISPER_HOST", "0.0.0.0")
PORT = int(os.getenv("WHISPER_PORT", "8100"))
MIN_CHUNK_SEC = float(os.getenv("WHISPER_MIN_CHUNK", "1.0"))
LOG_LEVEL = os.getenv("WHISPER_LOG_LEVEL", "INFO")
logger.setLevel(LOG_LEVEL)

# ── ASR 모델 로드 (서버 시작 시 1회) ──────────────────────────────────────────
import argparse

def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_shared_args(parser)
    args, _ = parser.parse_known_args()
    args = override_args_with_env(args)
    return args

args = _build_args()
logger.info(f"Whisper 모델 로드 중: model={args.model} backend={args.backend} lan={args.lan}")
_asr_singleton, _ = asr_factory(args)
logger.info("Whisper 모델 로드 완료")


# ── PCM 바이트 → float32 numpy ────────────────────────────────────────────────
def pcm_to_float32(raw_bytes: bytes) -> np.ndarray:
    sf = soundfile.SoundFile(
        io.BytesIO(raw_bytes),
        channels=1,
        endian="LITTLE",
        samplerate=SAMPLING_RATE,
        subtype="PCM_16",
        format="RAW",
    )
    audio, _ = librosa.load(sf, sr=SAMPLING_RATE, dtype=np.float32)
    return audio


# ── 세션별 ASR 프로세서 팩토리 ────────────────────────────────────────────────
def _make_online_processor():
    trimming = (args.buffer_trimming, args.buffer_trimming_sec)
    if args.vac:
        return VACOnlineASRProcessor(
            args.min_chunk_size,
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
async def asr_handler(websocket, path: str):
    params = parse_qs(urlparse(path).query)
    session_id = (params.get("session_id") or ["unknown"])[0]
    logger.info(f"[{session_id}] 연결됨")

    online = _make_online_processor()
    online.init()
    manager = ASRManager()

    min_bytes = int(MIN_CHUNK_SEC * SAMPLING_RATE * 2)  # int16 = 2 bytes/sample
    buffer = bytearray()

    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            buffer.extend(message)

            # min_chunk 이상 누적되면 추론
            while len(buffer) >= min_bytes:
                chunk_bytes = bytes(buffer[:min_bytes])
                buffer = buffer[min_bytes:]

                try:
                    audio = pcm_to_float32(chunk_bytes)
                except Exception as e:
                    logger.warning(f"[{session_id}] PCM 변환 오류: {e}")
                    continue

                online.insert_audio_chunk(audio)
                seg = online.process_iter()

                # seg = (beg, end, text)
                is_final = False
                result = manager.handle(seg, is_final)
                if result:
                    logger.info(f"[{session_id}] STT: {result!r}")
                    await websocket.send(result)

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"[{session_id}] 정상 종료")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"[{session_id}] 비정상 종료: {e}")
    except Exception as e:
        logger.error(f"[{session_id}] 오류: {e}", exc_info=True)
    finally:
        # 남은 버퍼 flush
        if len(buffer) > 0:
            try:
                audio = pcm_to_float32(bytes(buffer))
                online.insert_audio_chunk(audio)
                seg = online.finish()
                result = manager.handle(seg, is_final=True)
                if result:
                    await websocket.send(result)
            except Exception:
                pass
        logger.info(f"[{session_id}] 핸들러 종료")


# ── 진입점 ────────────────────────────────────────────────────────────────────
async def main():
    logger.info(f"npc_whisper ws_server 시작: ws://{HOST}:{PORT}/asr")
    async with websockets.serve(asr_handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
