# online_processor.py
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []
        self.last_commited_time = 0
        self.last_commited_word = None
        self.logfile = logfile

    def insert(self, new, offset):
        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if self.new:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1 and self.commited_in_buffer:
                cn = len(self.commited_in_buffer)
                nn = len(self.new)
                for i in range(1, min(min(cn, nn), 5) + 1):
                    c = " ".join([self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1])
                    tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                    if c == tail:
                        for _ in range(i):
                            self.new.pop(0)
                        logger.debug(f"removing last {i} words (n-gram dedup)")
                        break

    def flush(self):
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if not self.buffer:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break

        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile
        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming
        self.init()

    def init(self, offset=None):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0.0
        if offset is not None:
            self.buffer_time_offset = float(offset)
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio: np.ndarray):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1

        p = [t for _, _, t in self.commited[:k]]
        prompt = []
        l = 0
        while p and l < 200:
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)

        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _, _, t in non_prompt)

    def process_iter(self):
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}s from {self.buffer_time_offset:2.2f}")

        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        committed_words = self.transcript_buffer.flush()
        self.commited.extend(committed_words)

        if committed_words and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer) / self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        s = self.buffer_trimming_sec if self.buffer_trimming_way == "segment" else 30
        if len(self.audio_buffer) / self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

        return self.to_flush(committed_words)

    def chunk_completed_sentence(self):
        if not self.commited or self.tokenizer is None:
            return
        sents = self.words_to_sentences(self.commited)
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        chunk_at = sents[-2][1]
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if not self.commited:
            return
        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                self.chunk_at(e)

    def chunk_at(self, time: float):
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds * self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        self.buffer_time_offset += len(self.audio_buffer) / self.SAMPLING_RATE
        return f

    def to_flush(self, sents, sep=None, offset=0):
        if sep is None:
            sep = self.asr.sep
        text = sep.join(s[2] for s in sents)
        if not sents:
            return (None, None, "")
        b = offset + sents[0][0]
        e = offset + sents[-1][1]
        return (b, e, text)