# vad.py
import sys
import numpy as np
import logging

from online_processor import OnlineASRProcessor

logger = logging.getLogger(__name__)

class VACOnlineASRProcessor:
    """
    VAC wrapper:
      - runs Silero VAD iterator
      - feeds voice chunks into OnlineASRProcessor
      - emits final immediately when end-of-voice detected
    """

    SAMPLING_RATE = 16000

    def __init__(self, online_chunk_size: float, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(asr, tokenizer, buffer_trimming=buffer_trimming, logfile=logfile)

        import torch
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)

        self.logfile = logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status = None  # "voice" | "nonvoice" | None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_offset = 0  # in frames

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([], dtype=np.float32)

    def insert_audio_chunk(self, audio: np.ndarray):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0] - self.buffer_offset

            if "start" in res and "end" not in res:
                self.status = "voice"
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()

            elif "end" in res and "start" not in res:
                self.status = "nonvoice"
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()

            else:
                beg = res["start"] - self.buffer_offset
                end = res["end"] - self.buffer_offset
                self.status = "nonvoice"
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg + self.buffer_offset) / self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()

        else:
            if self.status == "voice":
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                # keep last 1s for possible VAD start detection
                self.buffer_offset += max(0, len(self.audio_buffer) - self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()

        if self.current_online_chunk_buffer_size > self.SAMPLING_RATE * self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            return self.online.process_iter()

        return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret