# asr_backends.py
import sys
import io
import math
import logging
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

class ASRBase:
    """
    Contract:
      - transcribe(audio: np.ndarray, init_prompt: str) -> backend result
      - ts_words(result) -> List[(start, end, token)]
      - segments_end_ts(result) -> List[end_ts]
    """
    sep = " "

    def __init__(self, lan: str, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.original_language = None if lan == "auto" else lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        raise NotImplementedError

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        raise NotImplementedError

    def ts_words(self, res):
        raise NotImplementedError

    def segments_end_ts(self, res):
        raise NotImplementedError

    def use_vad(self):
        raise NotImplementedError

    def set_translate_task(self):
        raise NotImplementedError


class WhisperTimestampedASR(ASRBase):
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("WhisperTimestampedASR: model_dir ignored (not implemented).")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        return self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            verbose=None,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )

    def ts_words(self, r):
        out = []
        for s in r["segments"]:
            for w in s["words"]:
                out.append((w["start"], w["end"], w["text"]))
        return out

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug("FasterWhisperASR: Loading from model_dir, ignoring modelsize/cache_dir.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir must be set")

        import os
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        return WhisperModel(
            model_size_or_path,
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
        )

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments):
        out = []
        for seg in segments:
            if getattr(seg, "no_speech_prob", 0.0) > 0.9:
                continue
            for w in seg.words:
                out.append((w.start, w.end, w.word))
        return out

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx

        if model_dir is not None:
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
        else:
            raise ValueError("modelsize or model_dir must be set")

        self.model_size_or_path = model_size_or_path

        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)  # preload
        return transcribe

    def translate_model_name(self, model_name: str) -> str:
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx",
        }
        if model_name not in model_mapping:
            raise ValueError(f"Unsupported MLX model name: {model_name}")
        return model_mapping[model_name]

    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs,
        )
        return segments.get("segments", [])

    def ts_words(self, segments):
        return [
            (w["start"], w["end"], w["word"])
            for seg in segments
            for w in seg.get("words", [])
            if seg.get("no_speech_prob", 0.0) <= 0.9
        ]

    def segments_end_ts(self, res):
        return [s["end"] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """
    OpenAI Whisper API backend
    - transcribe() returns OpenAI verbose_json object
    """
    sep = " "

    def __init__(self, lan="auto", temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature

        from openai import OpenAI
        self.client = OpenAI()

        self.use_vad_opt = False
        self.task = "transcribe"
        self.transcribed_seconds = 0

    def load_model(self, *args, **kwargs):
        return None

    def transcribe(self, audio_data: np.ndarray, init_prompt: str = ""):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        self.transcribed_seconds += math.ceil(len(audio_data) / 16000)

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"],
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if init_prompt:
            params["prompt"] = init_prompt

        proc = self.client.audio.translations if self.task == "translate" else self.client.audio.transcriptions
        return proc.create(**params)

    def ts_words(self, segments):
        no_speech = []
        if self.use_vad_opt:
            for seg in segments.segments:
                if seg.get("no_speech_prob", 0.0) > 0.8:
                    no_speech.append((seg.get("start"), seg.get("end")))

        out = []
        for w in segments.words:
            st, ed = w.start, w.end
            if any(a <= st <= b for a, b in no_speech):
                continue
            out.append((st, ed, w.word))
        return out

    def segments_end_ts(self, res):
        return [w.end for w in res.words]

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"