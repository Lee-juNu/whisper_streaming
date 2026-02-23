#!/usr/bin/env python3
"""
whisper_server.py

- Replaces `from whisper_online import *` with the split modules:
  - audio.py
  - asr_backends.py
  - online_processor.py
  - vad.py
  - manager.py (optional; not required for basic server)

- Keeps your protocol:
  - Client sends RAW PCM16 mono 16kHz bytes (no WAV header)
  - Server buffers until >= min_chunk_size seconds
  - Server returns: "beg_ms end_ms text" per confirmed segment
  - Avoids sending identical line twice (line_packet)

Notes:
- FIXED bug: `online.process_iter()` -> `self.online_asr_proc.process_iter()`
- Requires: line_packet.py and silero_vad_iterator.py in import path if using --vac
"""

import sys
import os
import io
import math
import socket
import argparse
import logging

import numpy as np
import librosa
import soundfile

from dotenv import load_dotenv

from manager import ASRManager
load_dotenv()

from audio import load_audio_chunk, SAMPLING_RATE
from asr_backends import (
    FasterWhisperASR,
    WhisperTimestampedASR,
    MLXWhisper,
    OpenaiApiASR,
)
from online_processor import OnlineASRProcessor
from vad import VACOnlineASRProcessor

import line_packet

logger = logging.getLogger(__name__)


# -----------------------------
# Args / Env
# -----------------------------

def add_shared_args(parser: argparse.ArgumentParser):
    parser.add_argument('--min-chunk-size', type=float, default=1.0,
                        help='Minimum audio chunk size in seconds. Server waits until >= this duration of audio is accumulated.')
    parser.add_argument('--model', type=str, default='large-v2',
                        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),
                        help="Whisper model name (default: large-v2).")
    parser.add_argument('--model_cache_dir', type=str, default=None,
                        help="Directory for model downloads / cache.")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Directory containing a pre-downloaded model. Overrides --model/--model_cache_dir.")
    parser.add_argument('--lan', '--language', type=str, default='auto',
                        help="Source language code (e.g., en, ja) or 'auto'.")
    parser.add_argument('--task', type=str, default='transcribe',
                        choices=["transcribe", "translate"],
                        help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper",
                        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],
                        help='ASR backend.')
    parser.add_argument('--vac', action="store_true", default=False,
                        help='Use VAC (Silero VAD controller). Requires torch + silero_vad_iterator.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04,
                        help='VAC input chunk size in seconds (client should ideally send near this size).')
    parser.add_argument('--vad', action="store_true", default=False,
                        help='Use backend-native VAD option when available.')
    parser.add_argument('--buffer_trimming', type=str, default="segment",
                        choices=["sentence", "segment"],
                        help='Trimming strategy in OnlineASRProcessor (tokenizer required for sentence).')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15,
                        help='Trim threshold in seconds.')
    parser.add_argument("-l", "--log-level", dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='DEBUG',
                        help="Log level.")


def override_args_with_env(args):
    env_map = {
        'model': 'WHISPER_MODEL',
        'lan': 'WHISPER_LANG',
        'backend': 'WHISPER_BACKEND',
        'min_chunk_size': 'WHISPER_MIN_CHUNK',
        'vad': 'WHISPER_VAD',
        'host': 'WHISPER_HOST',
        'port': 'WHISPER_PORT',
        'model_cache_dir': 'WHISPER_MODEL_CACHE_DIR',
        'model_dir': 'WHISPER_MODEL_DIR',
        'task': 'WHISPER_TASK',
        'vac': 'WHISPER_VAC',
        'vac_chunk_size': 'WHISPER_VAC_CHUNK',
        'buffer_trimming': 'WHISPER_TRIM',
        'buffer_trimming_sec': 'WHISPER_TRIM_SEC',
    }

    for arg, env in env_map.items():
        val = os.getenv(env)
        if val is None or not hasattr(args, arg):
            continue

        if arg in ('vad', 'vac'):
            setattr(args, arg, val.lower() in ('1', 'true', 'yes', 'y', 'on'))
        elif arg in ('port',):
            setattr(args, arg, int(val))
        elif arg in ('min_chunk_size', 'vac_chunk_size', 'buffer_trimming_sec'):
            setattr(args, arg, float(val))
        else:
            setattr(args, arg, val)

    return args


def set_logging(args):
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)


# -----------------------------
# ASR Factory (no whisper_online)
# -----------------------------

def asr_factory(args, logfile=sys.stderr):
    if args.backend == "openai-api":
        logger.debug("Using OpenAI API backend.")
        asr = OpenaiApiASR(lan=args.lan, logfile=logfile)
    else:
        if args.backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif args.backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR

        size = args.model
        logger.info(f"Loading Whisper model={size}, lan={args.lan}, backend={args.backend} ...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir, logfile=logfile)
        logger.info("Whisper model loaded.")

    if args.vad:
        logger.info("Enabling VAD option on backend (if supported).")
        asr.use_vad()

    if args.task == "translate":
        asr.set_translate_task()

    tokenizer = None  # keep server minimal; add tokenizer if you need sentence trimming
    trimming = (args.buffer_trimming, args.buffer_trimming_sec)

    if args.vac:
        online = VACOnlineASRProcessor(
            args.min_chunk_size,
            asr,
            tokenizer,
            buffer_trimming=trimming,
            logfile=logfile,
        )
        # server side min_chunk for waiting on audio is still args.min_chunk_size
        # VAC's internal "process_iter cadence" depends on online_chunk_size (args.min_chunk_size)
    else:
        online = OnlineASRProcessor(
            asr,
            tokenizer,
            buffer_trimming=trimming,
            logfile=logfile,
        )

    return asr, online


# -----------------------------
# Socket helpers
# -----------------------------

class Connection:
    """Wraps socket conn; provides line-based send/recv and raw-audio recv."""
    PACKET_SIZE = 32000 * 5 * 60  # bytes, ~5min @ 16kHz PCM16 mono (~32KB/s)

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.last_line = ""
        self.conn.setblocking(True)

    def send(self, line: str):
        # prevent sending identical line twice
        if line == self.last_line:
            return
        line_packet.send_one_line(self.conn, line)
        self.last_line = line

    def receive_lines(self):
        return line_packet.receive_lines(self.conn)

    def non_blocking_receive_audio(self):
        try:
            return self.conn.recv(self.PACKET_SIZE)
        except ConnectionResetError:
            return None


# -----------------------------
# Server processor
# -----------------------------

class ServerProcessor:
    def __init__(self, connection: Connection, online_asr_proc, min_chunk: float):
        self.connection = connection
        self.online_asr_proc = online_asr_proc
        self.min_chunk = float(min_chunk)
        self.manager = ASRManager()
        self.last_end_ms = None
        self.is_first = True

    def receive_audio_chunk(self):
        """
        Receive RAW PCM16LE mono 16kHz bytes until >= min_chunk seconds accumulated.
        Returns float32 numpy array @ 16kHz, or None if connection closed / insufficient first chunk.
        """
        out = []
        minlimit = int(self.min_chunk * SAMPLING_RATE)

        while sum(len(x) for x in out) < minlimit:
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                break

            # Read raw PCM16LE stream as SoundFile, then load via librosa to float32 @ 16kHz
            raw_f = soundfile.SoundFile(
                io.BytesIO(raw_bytes),
                channels=1,
                endian="LITTLE",
                samplerate=SAMPLING_RATE,
                subtype="PCM_16",
                format="RAW",
            )
            audio, _ = librosa.load(raw_f, sr=SAMPLING_RATE, dtype=np.float32)
            out.append(audio)

        if not out:
            return None

        conc = np.concatenate(out)
        if self.is_first and len(conc) < minlimit:
            # first chunk must reach minlimit, otherwise wait more
            return None

        self.is_first = False
        return conc

    def format_output_transcript(self, o):
        """
        o: (beg_sec|None, end_sec|None, text)
        Returns protocol line: "beg_ms end_ms text" or None.
        Ensures non-overlapping [beg,end] by max(beg, last_end).
        """
        if o[0] is None:
            logger.debug("No text in this segment")
            return None

        beg_ms = o[0] * 1000.0
        end_ms = o[1] * 1000.0

        if self.last_end_ms is not None:
            beg_ms = max(beg_ms, self.last_end_ms)

        self.last_end_ms = end_ms

        line = "%1.0f %1.0f %s" % (beg_ms, end_ms, o[2])
        print(line, flush=True, file=sys.stderr)
        return line

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            self.connection.send(msg)

    def process(self):
        # Handle one client connection
        self.online_asr_proc.init()

        while True:
            a = self.receive_audio_chunk()
            if a is None:
                break

            self.online_asr_proc.insert_audio_chunk(a)

            # FIX: use self.online_asr_proc, not global
            o = self.online_asr_proc.process_iter()
            llm_text = self.manager.handle(o, is_final=False)
            if llm_text:
                self.connection.send(f"LLM {llm_text}")
            try:
                self.send_result(o)
            except BrokenPipeError:
                logger.info("broken pipe -- connection closed?")
                break

        # Optional: flush remaining
        # o = self.online_asr_proc.finish()
        # self.send_result(o)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default=os.getenv("WHISPER_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WHISPER_PORT", "43007")))
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
                        help="Path to a speech wav file to warm up. Reads first 1s at 16kHz.")

    add_shared_args(parser)

    args = parser.parse_args()
    args = override_args_with_env(args)
    set_logging(args)

    # Create ASR & Online processor
    asr, online = asr_factory(args, logfile=sys.stderr)

    # Warmup
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("Warmup file not available. " + msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # Server loop
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info("Listening on %s" % str((args.host, args.port)))

        while True:
            conn, addr = s.accept()
            logger.info("Connected to client on %s" % (addr,))
            connection = Connection(conn)

            proc = ServerProcessor(connection, online, args.min_chunk_size)
            proc.process()

            conn.close()
            logger.info("Connection to client closed")


if __name__ == "__main__":
    main()