import sys, os, argparse, logging, socket
from dotenv import load_dotenv
from audio import load_audio_chunk, SAMPLING_RATE
from asr_backends import (
    FasterWhisperASR,
    WhisperTimestampedASR,
    MLXWhisper,
    OpenaiApiASR,
)
from online_processor import OnlineASRProcessor
from vad import VACOnlineASRProcessor
from connection import Connection
from threaded_processor import ThreadedServerProcessor

logger = logging.getLogger(__name__)
load_dotenv()

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
        'threshold': 'WHISPER_THRESHOLD',
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
        elif arg in ('min_chunk_size', 'vac_chunk_size', 'buffer_trimming_sec', 'threshold'):
            setattr(args, arg, float(val))
        else:
            setattr(args, arg, val)
    return args

def set_logging(args):
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)

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
    tokenizer = None
    trimming = (args.buffer_trimming, args.buffer_trimming_sec)
    if args.vac:
        online = VACOnlineASRProcessor(
            args.min_chunk_size,
            asr,
            tokenizer,
            buffer_trimming=trimming,
            logfile=logfile,
            logger=logger
        )
    else:
        online = OnlineASRProcessor(
            asr,
            tokenizer,
            buffer_trimming=trimming,
            logfile=logfile,
        )
    return asr, online

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
    # threshold 값이 없으면 기본값 사용
    threshold = getattr(args, 'threshold', 0.01)
    asr, online = asr_factory(args, logfile=sys.stderr)
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
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((args.host, args.port))
        s.listen(1)
        logger.info("Listening on %s" % str((args.host, args.port)))
        while True:
            conn, addr = s.accept()
            logger.info("Connected to client on %s" % (addr,))
            connection = Connection(conn)
            proc = ThreadedServerProcessor(connection, online, args.min_chunk_size, threshold=threshold)
            proc.process()
            conn.close()
            logger.info("Connection to client closed")

if __name__ == "__main__":
    main()
