# cli_simulator.py
#!/usr/bin/env python3
import sys
import time
import logging
import argparse

from audio import load_audio, load_audio_chunk, SAMPLING_RATE
from manager import ASRManager
from online_processor import OnlineASRProcessor
from vad import VACOnlineASRProcessor
from asr_backends import (
    WhisperTimestampedASR,
    FasterWhisperASR,
    MLXWhisper,
    OpenaiApiASR,
)

logger = logging.getLogger(__name__)

WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

def create_tokenizer(lan: str):
    assert lan in WHISPER_LANG_CODES, "unsupported lang code"

    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        lan = None

    from wtpsplit import WtP
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()

def add_shared_args(p: argparse.ArgumentParser):
    p.add_argument("--min-chunk-size", type=float, default=1.0)
    p.add_argument("--model", type=str, default="large-v2",
                   choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","))
    p.add_argument("--model_cache_dir", type=str, default=None)
    p.add_argument("--model_dir", type=str, default=None)
    p.add_argument("--lan", "--language", type=str, default="auto")
    p.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--backend", type=str, default="faster-whisper",
                   choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"])
    p.add_argument("--vac", action="store_true", default=False)
    p.add_argument("--vac-chunk-size", type=float, default=0.04)
    p.add_argument("--vad", action="store_true", default=False)
    p.add_argument("--buffer_trimming", type=str, default="segment", choices=["sentence", "segment"])
    p.add_argument("--buffer_trimming_sec", type=float, default=15)
    p.add_argument("-l", "--log-level", dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="DEBUG")

def set_logging(level: str):
    logging.basicConfig(format="%(levelname)s\t%(message)s")
    logger.setLevel(level)

def asr_factory(args):
    if args.backend == "openai-api":
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if args.backend == "faster-whisper":
            cls = FasterWhisperASR
        elif args.backend == "mlx-whisper":
            cls = MLXWhisper
        else:
            cls = WhisperTimestampedASR

        asr = cls(modelsize=args.model, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)

    if args.vad:
        asr.use_vad()

    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"
    else:
        tgt_language = args.lan

    tokenizer = create_tokenizer(tgt_language) if args.buffer_trimming == "sentence" else None
    trimming = (args.buffer_trimming, args.buffer_trimming_sec)

    if args.vac:
        proc = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, buffer_trimming=trimming, logfile=sys.stderr)
        min_chunk = args.vac_chunk_size
    else:
        proc = OnlineASRProcessor(asr, tokenizer, buffer_trimming=trimming, logfile=sys.stderr)
        min_chunk = args.min_chunk_size

    return asr, proc, min_chunk

def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio_path", type=str)
    add_shared_args(p)
    p.add_argument("--start_at", type=float, default=0.0)
    p.add_argument("--offline", action="store_true", default=False)
    p.add_argument("--comp_unaware", action="store_true", default=False)
    args = p.parse_args()

    if args.offline and args.comp_unaware:
        raise SystemExit("invalid: cannot use both --offline and --comp_unaware")

    set_logging(args.log_level)

    duration = len(load_audio(args.audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, processor, min_chunk = asr_factory(args)

    # warmup
    a = load_audio_chunk(args.audio_path, 0, 1)
    asr.transcribe(a)

    def on_text(text, beg, end):
        now_ms = (time.time() - start) * 1000
        print("%1.4f %1.0f %1.0f %s" % (now_ms, beg * 1000, end * 1000, text), flush=True)

    manager = ASRManager(processor, on_text=on_text)

    beg = args.start_at
    start = time.time() - beg

    if args.offline:
        a = load_audio(args.audio_path)
        manager.push_audio(a)
        manager.finish()
        return

    if args.comp_unaware:
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(args.audio_path, beg, end)
            manager.push_audio(a)

            if end >= duration:
                break
            beg = end
            end = min(duration, end + min_chunk)

        manager.finish()
        return

    # simultaneous mode
    end = 0.0
    while True:
        now = time.time() - start
        if now < end + min_chunk:
            time.sleep(min_chunk + end - now)

        end = time.time() - start
        a = load_audio_chunk(args.audio_path, beg, end)
        beg = end

        manager.push_audio(a)

        if end >= duration:
            break

    manager.finish()

if __name__ == "__main__":
    main()