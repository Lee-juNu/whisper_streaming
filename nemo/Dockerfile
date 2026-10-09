# npc_nemo — Nemotron ASR 서버(NVIDIA NeMo-Speech.cpp)를 **이미지에 통째로 굽는다.**
#
# 왜 bind mount 가 아니라 이미지인가: 재부팅 직후 Docker Desktop 이 이 WSL 배포판의
# 파일공유를 아직 못 받은 시점에 엔진이 컨테이너를 되살리면, 이 리포 경로의 bind 는
# 빈 디렉토리로 보인다. 실행 파일이 그 안에 있으면 `exec: no such file` 로 runc create 가
# 깨지고, 프로세스가 시작조차 안 하므로 restart 정책은 재시도하지 않는다 — 2026-10-04 재부팅
# 뒤 61시간 동안 STT 가 조용히 죽어 있었던 원인(npc_whisper 는 nemo 를 기다리다 1211회 재시작).
# 이미지 안에 있으면 bind 가 없으니 레이스 자체가 없다. syrinx(voices.json 씨앗)도 같은 길을 갔다.
#
# 원본은 둘 다 고정 리비전이고 SHA-256 을 검증한다(comparison/local/download-manifest.json 과 동일).
# 벤치(comparison/)는 같은 파일을 로컬에 두고 쓰지만 이 빌드는 그 디렉토리에 기대지 않는다 —
# comparison/local 은 gitignore 라 깨끗한 체크아웃에는 없다.
#
# 베이스가 ubuntu 인 이유: 번들이 libcudart·libcublas 를 동봉하고(bin/nemo-speech 의
# RUNPATH=$ORIGIN/../lib), libcuda 는 nvidia 런타임이 드라이버에서 넣어 준다. CUDA 이미지가 필요 없다.
FROM ubuntu:22.04

ARG NEMO_SPEECH_URL=https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/nemo-speech-0.1.0-linux-x86_64-cuda.tar.gz
ARG NEMO_SPEECH_SHA256=e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147
# Nemotron 3.5 ASR streaming 0.6B, Q8_0. 3070 8GB 에서 whisper small 대신 올리는 단 하나의 STT 모델.
ARG ASR_MODEL_URL=https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/resolve/ea30d66debe3740a08b573244286791d423d6b3e/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf
ARG ASR_MODEL_SHA256=3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1
ARG ASR_MODEL_FILE=nemotron-3.5-asr-streaming-0.6b.q8_0.gguf

ENV DEBIAN_FRONTEND=noninteractive
# curl 은 다운로드와 compose 헬스체크(/ready) 둘 다에 쓴다.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 런타임(107MB). 아카이브 최상위 디렉토리 하나를 벗겨 /opt/nemo/{bin,lib,...} 로 둔다.
RUN set -eux; \
    curl -fL --retry 3 --retry-delay 5 -o /tmp/nemo-speech.tar.gz "$NEMO_SPEECH_URL"; \
    echo "$NEMO_SPEECH_SHA256  /tmp/nemo-speech.tar.gz" | sha256sum -c -; \
    mkdir -p /opt/nemo; \
    tar -xzf /tmp/nemo-speech.tar.gz -C /opt/nemo --strip-components=1; \
    rm /tmp/nemo-speech.tar.gz; \
    test -x /opt/nemo/bin/nemo-speech

# 모델(742MB). 모델만 바꿀 때 런타임 레이어는 캐시에 남도록 뒤에 둔다.
RUN set -eux; \
    mkdir -p /models; \
    curl -fL --retry 3 --retry-delay 5 -o "/models/$ASR_MODEL_FILE" "$ASR_MODEL_URL"; \
    echo "$ASR_MODEL_SHA256  /models/$ASR_MODEL_FILE" | sha256sum -c -

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    ASR_MODEL_PATH=/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf

EXPOSE 18101

# 실제 인자는 compose 의 command 가 준다(모델 경로·포트·배칭 등 운용 노브는 거기서 한눈에 본다).
ENTRYPOINT ["/opt/nemo/bin/nemo-speech"]
CMD ["serve", "--asr.model.path", "/models/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf", "--device", "cuda:0", "--host", "0.0.0.0", "--port", "18101", "--no-ui"]
