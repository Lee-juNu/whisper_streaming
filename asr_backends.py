# asr_backends.py
import os
import sys
import io
import math
import logging
import json
import time
import uuid
import wave
import urllib.error
import urllib.request
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# 세그먼트 모드(발화 단위 전사)의 beam size. 스트리밍은 같은 오디오를 몇 번씩
# 다시 추론하므로 beam=1 이 맞지만, 발화당 1회인 쪽은 정확도를 사는 편이 남는다.
UTTERANCE_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

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

    def transcribe_utterance(self, audio: np.ndarray, init_prompt: str = ""):
        """발화 **한 덩어리**를 한 번에 전사한다(세그먼트 모드).

        스트리밍용 transcribe() 와 목적이 다르다. 저쪽은 같은 오디오를 여러 번
        재추론해 LocalAgreement 로 확정하므로 속도(beam=1)와 단어 타임스탬프가
        필요하지만, 이쪽은 발화당 딱 한 번이라 **정확도 쪽에 예산을 쓴다**.
        기본 구현은 transcribe() 로 위임한다 — 백엔드가 따로 최적화할 수 있다.
        """
        return self.transcribe(audio, init_prompt=init_prompt)

    def utterance_segments(self, res):
        """전사 결과를 (text, no_speech_prob, avg_logprob) 목록으로 정규화한다.

        백엔드마다 결과 모양이 다르다(dict / 객체 / {"segments": [...]}).
        세그먼트 모드의 환각 필터가 이 세 값만 보므로 여기서 한 번에 흡수한다.
        """
        segs = res.get("segments", []) if isinstance(res, dict) else res
        out = []
        for s in segs or []:
            if isinstance(s, dict):
                text = s.get("text", "")
                nsp  = s.get("no_speech_prob", 0.0)
                alp  = s.get("avg_logprob", 0.0)
            else:
                text = getattr(s, "text", "")
                nsp  = getattr(s, "no_speech_prob", 0.0)
                alp  = getattr(s, "avg_logprob", 0.0)
            out.append((text or "", float(nsp or 0.0), float(alp or 0.0)))
        return out

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
            beam_size=1,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def transcribe_utterance(self, audio: np.ndarray, init_prompt: str = ""):
        """발화 한 덩어리 전용. 스트리밍 경로와 세 가지가 다르다.

          - ``beam_size`` 를 올린다(기본 5). 발화당 1회라 그만큼 쓸 수 있다.
          - ``word_timestamps=False``. 단어 단위 확정을 하지 않으므로 순수 비용이다.
          - ``vad_filter=True``. 발화 앞뒤의 무음·잡음을 인코더에 넣지 않는다 —
            세그먼트 모드에서 **환각을 막는 1차 방어선**이다.

        ``condition_on_previous_text`` 는 끈다. 호출이 독립적이라 이 옵션은
        같은 호출 안의 30초 창들 사이에서만 의미가 있는데, 거기서 한 번 헛나가면
        뒤가 통째로 끌려간다. 문맥은 init_prompt 로 **우리가 고른 것만** 넣는다.
        """
        kwargs = dict(self.transcribe_kargs)
        kwargs.pop("vad_filter", None)
        segments, _info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt or None,
            beam_size=UTTERANCE_BEAM_SIZE,
            word_timestamps=False,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            **kwargs,
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


class NemotronASR(ASRBase):
    """NVIDIA NeMo-Speech.cpp(`nemo-speech serve`)의 HTTP 파일 전사를 쓰는 백엔드.

    **세그먼트 모드 전용**이다. 모델은 이 프로세스가 아니라 별도 서버(compose 의
    ``nemo`` 서비스)가 들고 있고, 여기서는 발화 WAV 를 ``POST /v1/audio/transcriptions``
    로 보내 텍스트만 받는다. 스트리밍(LocalAgreement) 경로는 같은 오디오를 여러 번
    재추론하는 whisper 전제라 지원하지 않는다 — 진짜 스트리밍이 필요하면
    ``/v1/realtime`` WebSocket 을 따로 붙여야 한다.

    whisper 와 다른 점 셋:
      - ``init_prompt`` 를 **무시한다**. nemo-speech 의 ``prompt`` 는 문맥이 아니라
        「boost 10 으로 밀어 올릴 구절 하나」라, 앞 발화를 넣으면 그 문장을 통째로
        우대한다. 고유명사 바이어스는 ``speech_contexts`` 로 따로 붙일 것.
      - ``no_speech_prob``·``avg_logprob`` 가 없다. 0.0 으로 채워 ws_server 의
        두 필터가 항상 통과하게 한다(블록리스트·루프 판정은 그대로 산다).
      - HTTP 파일 경로는 서버의 **offline 모드**(attention-right=3)로 돈다.
        160ms 스트리밍 수치가 아니라 발화당 1회 전사의 수치가 나온다.

    환경변수: ``NEMO_URL``(기본 http://nemo:18101) · ``NEMO_LANGUAGE``(기본은 WHISPER_LANG
    을 ja→ja-JP 식으로 변환) · ``NEMO_VERBATIM``(기본 true, ITN 끔) ·
    ``NEMO_PUNCTUATION``(기본 true) · ``NEMO_READY_TIMEOUT_SEC``(기본 0 = 무한정 대기;
    양수면 그 초 안에 /ready 가 안 오면 RuntimeError) · ``NEMO_HTTP_TIMEOUT_SEC``(기본 30).
    """
    sep = ""

    # whisper 식 2글자 코드 → nemo-speech 가 받는 로캘. 없는 것은 그대로 넘긴다.
    _LANG_MAP = {
        "ja": "ja-JP", "ko": "ko-KR", "en": "en-US", "zh": "zh-CN",
        "de": "de-DE", "es": "es-ES", "fr": "fr-FR", "it": "it-IT", "pt": "pt-BR", "ru": "ru-RU",
    }

    def __init__(self, lan="auto", base_url="http://nemo:18101", logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.base_url = base_url.rstrip("/")
        lan = (lan or "auto").strip()
        self.original_language = None if lan == "auto" else lan
        env_lang = os.getenv("NEMO_LANGUAGE", "").strip()
        if env_lang:
            self.language = env_lang
        elif self.original_language:
            self.language = self._LANG_MAP.get(self.original_language, self.original_language)
        else:
            self.language = None
        self.verbatim        = os.getenv("NEMO_VERBATIM", "true").strip().lower()
        self.punctuation     = os.getenv("NEMO_PUNCTUATION", "true").strip().lower()
        self.ready_timeout_s = float(os.getenv("NEMO_READY_TIMEOUT_SEC", "0"))
        self.http_timeout_s  = float(os.getenv("NEMO_HTTP_TIMEOUT_SEC", "30"))
        self.model = self.load_model()

    # ── 서버 준비 대기 ──────────────────────────────────────────────────────
    def _get_json(self, path: str, timeout: float = 5.0):
        with urllib.request.urlopen(self.base_url + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        """모델을 올리지 않는다. nemo-speech 의 ``/ready`` 가 true 가 될 때까지 기다리고
        ``/v1/models`` 의 모델 ID 를 돌려준다.

        compose 의 depends_on(service_healthy)이 첫 겹이고 여기가 두 번째 겹이다 — 엔진이
        재부팅 뒤 컨테이너를 되살릴 때는 depends_on 순서가 보장되지 않는다.
        기본은 **무한정 대기**다. 상한을 두고 exit 하면 nemo 가 죽어 있는 동안 restart 정책이
        3분마다 되살려 재시작 횟수만 쌓인다(2026-10-04~06, 1211회). 기다리는 동안 8100 을 안 열므로
        compose healthcheck 가 unhealthy 로 드러낸다. 로그는 30초에 한 줄만 남긴다."""
        deadline = (time.monotonic() + self.ready_timeout_s) if self.ready_timeout_s > 0 else None
        last_err = None
        last_log = None
        while True:
            try:
                ready = self._get_json("/ready")
                if isinstance(ready, dict) and ready.get("ready") is True:
                    break
                last_err = f"ready={ready!r}"
            except Exception as e:  # 연결 거부·타임아웃·JSON 오류 전부 재시도
                last_err = repr(e)
            if deadline is not None and time.monotonic() >= deadline:
                raise RuntimeError(
                    f"nemo-speech 가 {self.ready_timeout_s:.0f}s 안에 준비되지 않음 "
                    f"({self.base_url}): {last_err}"
                )
            now = time.monotonic()
            if last_log is None or now - last_log >= 30.0:
                logger.info(f"NemotronASR: {self.base_url} 대기 중 ({last_err})")
                last_log = now
            time.sleep(1.0)

        ids = []
        try:
            models = self._get_json("/v1/models")
            ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
        except Exception as e:
            logger.warning(f"NemotronASR: /v1/models 조회 실패 (계속 진행): {e!r}")
        logger.info(f"NemotronASR: {self.base_url} 준비됨 models={ids} language={self.language}")
        return ids[0] if ids else None

    # ── 전사 ────────────────────────────────────────────────────────────────
    @staticmethod
    def _wav_bytes(audio: np.ndarray) -> bytes:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        return buf.getvalue()

    def transcribe_utterance(self, audio: np.ndarray, init_prompt: str = ""):
        """발화 WAV 하나를 multipart 로 보내고 ``{"text": ...}`` 를 받는다.
        ``init_prompt`` 는 의도적으로 쓰지 않는다(클래스 설명 참조)."""
        fields = {
            "response_format": "json",
            "verbatim": self.verbatim,
            "automatic_punctuation": self.punctuation,
        }
        if self.language:
            fields["language"] = self.language

        boundary = "----npcwhisper" + uuid.uuid4().hex
        body = bytearray()
        for k, v in fields.items():
            body += (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
            ).encode("utf-8")
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"utterance.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8")
        body += self._wav_bytes(audio)
        body += f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            self.base_url + "/v1/audio/transcriptions",
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.http_timeout_s) as r:
                resp = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"nemo-speech HTTP {e.code}: {detail}") from e

        text = resp.get("text") if isinstance(resp, dict) else None
        if not isinstance(text, str):
            raise RuntimeError(f"nemo-speech 응답에 text 가 없음: {str(resp)[:200]}")
        # ws_server 의 필터가 보는 세 값으로 정규화. 확률 둘은 nemo 가 주지 않으므로 통과값.
        return {"segments": [{"text": text, "no_speech_prob": 0.0, "avg_logprob": 0.0}]}

    # ── 스트리밍 계약은 지원하지 않는다 ─────────────────────────────────────
    def transcribe(self, audio: np.ndarray, init_prompt: str = ""):
        raise NotImplementedError("NemotronASR 는 세그먼트 모드 전용이다 (WHISPER_MODE=stream 미지원)")

    def ts_words(self, res):
        raise NotImplementedError("NemotronASR 는 단어 타임스탬프를 쓰는 스트리밍 경로를 지원하지 않는다")

    def segments_end_ts(self, res):
        raise NotImplementedError("NemotronASR 는 스트리밍 경로를 지원하지 않는다")

    def use_vad(self):
        logger.info("NemotronASR: WHISPER_VAD 무시 — 서버 측 VAD 마스킹은 nemo-speech 기동 인자로 켠다")

    def set_translate_task(self):
        raise NotImplementedError("NemotronASR: translate 미지원")
