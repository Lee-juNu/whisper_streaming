# GPU 스모크 테스트 (단일 모델, 순차 실행)

공식 모델/런타임 설치와 순차 GPU 스모크는 **완료·검증됨**(Codex 실행, 2026-10-01).
샘플은 공개 일본어 음성의 앞 12초(`local/audio/ja_smoke_12s.wav`)만 사용합니다. 비공개 음성 없음.
**정확도 벤치마크가 아닙니다**(정답 없음, transcript가 비어있지 않은지만 확인).

## 설치 상태 (검증됨)

글로벌 설치·드라이버 변경 없음. Kotoba는 의존성이 고정된 기존 docker 이미지를 그대로 사용합니다.

| 항목 | 내용 |
|---|---|
| 장비 | RTX 3070 Laptop 8GB, 드라이버 556.12 (WSL2) |
| Kotoba v2 | `local/models/kotoba-whisper-v2`, revision `f44edd35…` (`lock.json`), 이미지 `whisper_streaming-whisper:latest` |
| Nemotron 모델 | `local/models/nemotron/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` |
| NeMo-Speech 런타임 | 공식 릴리스 바이너리 v0.1.0 (CUDA 12 아카이브), `local/nemo-runtime/nemo-speech-0.1.0-linux-x86_64-cuda/` |
| 릴리스 커밋 | `4f9676226f667d14608487df744f375db87127f8` — `lock.json`의 원래 소스 pin(`4c101…`)과 **다름**. 소스 빌드가 아니라 릴리스 바이너리이며, 실제 커밋은 릴리스 커밋입니다. |
| 아카이브 SHA-256 | `e68628f396489c98fb353e070efaea5bc4977409ae7734fce56c251a79e29147` |
| 모델 해시 | `local/download-manifest.json` 참조 |
| 샘플 음성 | `kotoba-tech/kotoba-whisper-v1.0-ggml` revision `bc0fb8704ab1108e06e3eaedeca1bf458ddbcd11`의 공개 샘플, 앞 12초 (`local/audio/sample_ja_speech.source.json`) |

## 원칙

- GPU에는 한 번에 1개 모델만. `BenchLock` 획득 후 실행.
- **GPU 0만 지원.** `--gpu`가 0이 아니면 아무것도 조회/실행하지 않고 거부. Kotoba는 `--gpus device=0`,
  Nemotron은 `--device cuda:0`, 자식에 `CUDA_VISIBLE_DEVICES=0`·`CUDA_DEVICE_ORDER=PCI_BUS_ID`, VRAM은 nvidia-smi index 0.
- `npc_whisper` 컨테이너, 다른 `asrbench-smoke-*` 컨테이너, 라이브 ASR로 보이는 프로세스가 있으면 **거부**.
  이 도구는 서비스를 정지/관리하지 않습니다. 이후 실행에서도 운영자가 직접 배타 윈도우를 확보해야 합니다.
- 자식 프로세스는 argv로 실행(shell 미사용), cwd는 항상 comparison 디렉터리(호출 위치와 무관).
  timeout·KeyboardInterrupt·예외 시 **자기 PID/자기 컨테이너만** kill 후 reap(`finally`에서 컨테이너 제거 확인).
- 실행 전후 GPU **디바이스 전체** memory.used(MiB, 모델 단독 할당량 아님)를 기록하고, 실행 전 값 + 100MiB 이내로
  돌아올 때까지 제한 시간 대기. 돌아오지 않으면 실패. 프로세스 reap·컨테이너 제거 여부도 기록.
- CPU 폴백 없음: Kotoba는 `device="cuda"`, `float16`, `local_files_only=True`로 로드하고 `engine.device == "cuda"`를 assert.
- 출력에 전체 커맨드라인은 포함하지 않음.

## 실행

```bash
cd comparison
python3 gpu_smoke.py --model kotoba --allow-gpu
python3 gpu_smoke.py --model nemotron --allow-gpu
```

네이티브 argv를 직접 줄 필요는 없습니다. Nemotron 기본 커맨드(절대 경로로 전개):

```
local/nemo-runtime/nemo-speech-0.1.0-linux-x86_64-cuda/bin/nemo-speech transcribe local/audio/ja_smoke_12s.wav \
  --model local/models/nemotron/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --device cuda:0 --language ja-JP \
  --format json --stream --no-batching --asr.streaming.rnnt_right_context 1
```

`--` 뒤에 argv를 직접 주는 경우에도 실행 전에 검증합니다: 바이너리가 `local/nemo-runtime/` 아래에 존재,
첫 인자 `transcribe`, `--device cuda:0` 정확히 1회(`--backend` 금지), `--model`은 `local/models/` 아래 존재하는
`.gguf`(다운로드 없음), `--format json` 정확히 1회.
성공 판정은 stdout을 JSON으로 파싱한 `text`가 비어있지 않고, stderr에 `backend=CUDA0`가 있을 때만입니다
(help 출력·빈 JSON·CPU 백엔드는 실패).

Kotoba 컨테이너는 `--gpus device=0 --network none`, comparison 디렉터리를 `/cmp`에 읽기 전용 마운트, entrypoint `python3.11`.
1회 실행 안에서 cold load → warmup 1회 → 측정 1회(제너레이터 소비, language=ja, beam=5).

## 최종 코드 재검증 (`results/gpu-final-20261001T052302Z/`)

두 테스트 모두 **SUCCESS**(순차 실행). 테스트 동안 Codex가 승인된 범위에서 `npc_whisper`만 일시 정지하고 finally에서 복구
(small 모델 로드 완료 및 8100/asr 서버 시작 로그 확인, GPU 1538MiB 관측). 이 도구 자체는 baseline을 정지하지 않습니다.

| 모델 | 시간 (12초 음성) | VRAM 전/피크/후 (디바이스 전체, MiB) |
|---|---|---|
| Kotoba v2 | cold load 8.961s, warmup 1.497s, recognition 0.832s | 859 / 2868 / 859 |
| Nemotron (streaming, backend=CUDA0, step 160ms) | 단계별 `unknown`; CUDA0 일본어 스트리밍 전사 성공 | 859 / 1920 / 859 |

Nemotron의 1.628s(`subprocess_wall_ms`)는 프로세스 기동·모델 로드·인식·종료를 모두 포함한 시간이며
**인식 지연이 아닙니다.** 네이티브 CLI가 단계를 분리해 보고하지 않으므로 cold/warmup/recognition은 `unknown`입니다.

## 테스트 (CPU mock, GPU/docker 불필요)

```bash
python3.12 -m unittest discover -s tests -v
```

## Nemotron 단계별 별도 측정

`results/nemotron-phases-20261001T052014Z/result.json`에 별도 localhost HTTP 시험을 저장했습니다.
공식 CUDA 런타임을 `--no-warmup`으로 새로 기동하여 `/v1/models`의 `device=cuda:0`를 확인했습니다.
프로세스 시작→HTTP 준비 **304.6ms**, 첫 일본어 요청(warmup) **730.7ms**, 같은 12초 파일의 다음 요청 **105.2ms**.
요청 시간은 로컬 HTTP 전송/전처리/인식/응답을 포함하며, 위 streaming CLI와 모드가 달라 직접 속도 비교하지 않습니다.
새 프로세스 모델 초기화 측정이며 OS 파일 캐시가 비었다는 뜻의 cold-disk benchmark는 아닙니다.
Kotoba cold_load에도 Python import 시간이 포함됩니다. 어느 숫자도 발화 종료→NPC 응답 지연이 아닙니다.

최종 stdlib 테스트 **33개 PASS**. 최종 GPU 실행은 양쪽 모두 종료 코드 0, 비어 있지 않은 일본어 전사,
프로세스 회수 및 VRAM 반환을 확인했습니다. Kotoba의 전용 컨테이너도 제거됐습니다.
100ms VRAM 샘플은 디바이스 전체 사용량이며, WSL은 Windows 쪽 프로세스별 GPU 귀속을 완전히 제공하지 않습니다.
확인된 baseline ASR을 정지하고 테스트 ASR을 하나씩 실행했으며 다른 그래픽 프로세스는 건드리지 않았습니다.
메모리/NPC 코드 배포, DB 변경, 커밋/푸시 없음. 모델·런타임·결과는 `.gitignore`로 제외됩니다.
공정한 CER/닉네임 정확도 및 실제 실시간 지연 비교는 사용자 검증 정답/타임스탬프가 있는 데이터셋으로 별도 수행해야 합니다.
