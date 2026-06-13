FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        curl \
        ca-certificates \
        build-essential \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# get-pip.py 로 python3.11 전용 pip 설치 (ensurepip 가 없으므로)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py \
    && rm /tmp/get-pip.py

WORKDIR /app

RUN python3.11 -m pip install --no-cache-dir \
    python-dotenv \
    numpy \
    librosa \
    soundfile \
    websockets \
    faster-whisper

RUN python3.11 -m pip install --no-cache-dir \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

COPY manager.py \
     audio.py \
     online_processor.py \
     vad.py \
     whisper_online.py \
     whisper_online_server.py \
     asr_backends.py \
     line_packet.py \
     silero_vad_iterator.py \
     threaded_processor.py \
     ws_server.py \
     ./

EXPOSE 8100

CMD ["python", "ws_server.py"]
