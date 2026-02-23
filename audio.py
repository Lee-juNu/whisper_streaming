# audio.py
import numpy as np
import librosa
from functools import lru_cache

SAMPLING_RATE = 16000

@lru_cache(10**6)
def load_audio(fname: str) -> np.ndarray:
    a, _ = librosa.load(fname, sr=SAMPLING_RATE, dtype=np.float32)
    return a

def load_audio_chunk(fname: str, beg: float, end: float) -> np.ndarray:
    audio = load_audio(fname)
    beg_s = int(beg * SAMPLING_RATE)
    end_s = int(end * SAMPLING_RATE)
    return audio[beg_s:end_s]