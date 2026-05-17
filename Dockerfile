# Whisper Streaming Dockerfile
# Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        git \
        && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements (if exists) or install inline
COPY .env ./
COPY whisper_online_server.py ./
COPY whisper_online.py ./
COPY line_packet.py ./
COPY silero_vad_iterator.py ./
COPY mic_to_tcp.py ./
COPY README.md ./

# Install python packages
RUN pip install --no-cache-dir python-dotenv librosa soundfile faster-whisper torch torchaudio

# Expose port (default from .env)
EXPOSE 43007

# Entrypoint: .env 적용하여 서버 실행
CMD ["python", "whisper_online_server.py"]
