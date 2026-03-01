import threading, queue, numpy as np, soundfile, io, logging, librosa
from manager import ASRManager

class ThreadedServerProcessor:
    def __init__(self, connection, online_asr_proc, min_chunk: float, threshold: float = 0.01, max_sec: float = 10.0):
        self.connection = connection
        self.online_asr_proc = online_asr_proc
        self.min_chunk = float(min_chunk)
        self.manager = ASRManager()
        self.last_end_ms = None
        self.llm_queue = []
        self.no_text_count = 0
        self.audio_queue = queue.Queue(maxsize=100)
        self.llm_request_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        self.threshold = threshold
        self.max_sec = max_sec
        self.logger = logging.getLogger(__name__)

    def producer(self):
        while not self.stop_event.is_set():
            raw_bytes = self.connection.non_blocking_receive_audio()
            if not raw_bytes:
                continue
            try:
                raw_f = soundfile.SoundFile(
                    io.BytesIO(raw_bytes),
                    channels=1,
                    endian="LITTLE",
                    samplerate=16000,
                    subtype="PCM_16",
                    format="RAW",
                )
                audio, _ = librosa.load(raw_f, sr=16000, dtype=np.float32)
                self.audio_queue.put(audio)
            except Exception as e:
                self.logger.error(f"Producer error: {e}")

    def consumer(self):
        import datetime, time
        buffer = []
        buffer_sec = 0.0
        last_audio_time = time.time()
        silence_flush_sec = 2.0  # 무음 지속 시간 (초)
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=1)
                last_audio_time = time.time()
                buffer.append(audio)
                buffer_sec += len(audio) / 16000
                trigger = None
                if self.is_silence(audio):
                    trigger = 'silence'
                elif buffer_sec < self.threshold:
                    trigger = 'threshold'
                elif buffer_sec > self.max_sec:
                    trigger = 'max_sec'
                if trigger:
                    conc = np.concatenate(buffer)
                    peak = np.max(np.abs(conc))
                    mean = np.mean(conc)
                    minv = np.min(conc)
                    maxv = np.max(conc)
                    energy = np.mean(conc ** 2)
                    with open('whisper_chunk.log', 'a', encoding='utf-8') as f:
                        f.write(f"[{datetime.datetime.now()}] trigger={trigger}, buffer_sec={buffer_sec:.3f}, buffer_len={len(buffer)}, peak={peak:.6f}, mean={mean:.6f}, min={minv:.6f}, max={maxv:.6f}, energy={energy:.6f}\n")
                    # VAD처럼, 말하는 도중에 무음이 감지되면 바로 끊어서 Whisper에 요청
                    if trigger == 'silence' and peak > 0.001:
                        self.online_asr_proc.insert_audio_chunk(conc)
                        o = self.online_asr_proc.process_iter()
                        self.send_result(o)
                    elif trigger == 'max_sec' and peak > 0.001:
                        self.online_asr_proc.insert_audio_chunk(conc)
                        o = self.online_asr_proc.process_iter()
                        self.send_result(o)
                    else:
                        self.logger.info(f"Skip Whisper request: peak={peak:.6f} (too low)")
                    buffer.clear()
                    buffer_sec = 0.0
            except queue.Empty:
                # audio가 일정 시간 안 들어오면 flush
                now = time.time()
                if buffer and (now - last_audio_time) > silence_flush_sec:
                    conc = np.concatenate(buffer)
                    peak = np.max(np.abs(conc))
                    mean = np.mean(conc)
                    minv = np.min(conc)
                    maxv = np.max(conc)
                    energy = np.mean(conc ** 2)
                    with open('whisper_chunk.log', 'a', encoding='utf-8') as f:
                        f.write(f"[{datetime.datetime.now()}] trigger=timeout_flush, buffer_sec={buffer_sec:.3f}, buffer_len={len(buffer)}, peak={peak:.6f}, mean={mean:.6f}, min={minv:.6f}, max={maxv:.6f}, energy={energy:.6f}\n")
                    if peak > 0.001:
                        self.online_asr_proc.insert_audio_chunk(conc)
                        o = self.online_asr_proc.process_iter()
                        self.send_result(o)
                    else:
                        self.logger.info(f"Skip Whisper request: peak={peak:.6f} (too low)")
                    buffer.clear()
                    buffer_sec = 0.0
                continue

    def is_silence(self, audio):
        return np.max(np.abs(audio)) < 0.02

    def send_result(self, o):
        msg_tuple = self.format_output_transcript(o)
        if msg_tuple is not None:
            msg, text = msg_tuple
            if "No text in this segment" in msg:
                if self.llm_queue:
                    self.no_text_count += 1
                    if self.no_text_count >= 5:
                        # Dify 요청 큐에 추가
                        self.llm_request_queue.put("\n".join(self.llm_queue))
                        self.llm_queue.clear()
                        self.no_text_count = 0
            else:
                self.llm_queue.append(text)
                self.no_text_count = 0

    def merge_segments(self, segments, max_gap_ms=500):
        merged = []
        buf = []
        last_end = None
        for beg, end, text in segments:
            if last_end is not None and beg - last_end <= max_gap_ms:
                buf.append(text)
            else:
                if buf:
                    merged.append(" ".join(buf))
                buf = [text]
            last_end = end
        if buf:
            merged.append(" ".join(buf))
        return merged

    def format_output_transcript(self, o):
        if o[0] is None:
            self.logger.debug("No text in this segment")
            self.no_text_count += 1
            if self.no_text_count > 5:
                self.llm_queue.clear()
                self.no_text_count = 0
            return None
        # o: (beg, end, text)
        # Try to parse text into segments for merging
        # Assume text is from consecutive segments, so we need to split by lines if possible
        # For now, treat as one segment
        beg_ms = o[0] * 1000.0
        end_ms = o[1] * 1000.0
        if self.last_end_ms is not None:
            beg_ms = max(beg_ms, self.last_end_ms)
        self.last_end_ms = end_ms
        # If you have a list of segments, merge here. For now, just output as is.
        line = "%1.0f %1.0f %s" % (beg_ms, end_ms, o[2])
        print(line, flush=True)
        return line, o[2]

    # send_to_llm는 더 이상 사용하지 않음. DifyThread에서 처리
    def dify_thread(self):
        while not self.stop_event.is_set():
            try:
                conversation = self.llm_request_queue.get(timeout=1)
                from llm_sender import send_to_dify
                result = send_to_dify(conversation)
                if result:
                    self.connection.send(f"{result}")
                print("[LLM.py] Dify API result:", result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[LLM.py] Error sending to Dify: {e}")

    def process(self):
        self.online_asr_proc.init()
        t1 = threading.Thread(target=self.producer)
        t2 = threading.Thread(target=self.consumer)
        t3 = threading.Thread(target=self.dify_thread)
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
