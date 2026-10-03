# asrbench — 일본어 ASR 비교 벤치마크 (faster-whisper small / Kotoba v2 / Nemotron 3.5 streaming)

운영 중인 `ws_server.py`와 완전히 분리된 비교 하네스입니다. `ws_server`는 **절대 import하지 않습니다**
(GPU 모델을 import 시점에 로드하기 때문). 코어(metrics/dataset/guards/mock/runner/CLI)는 Python 3.12 표준
라이브러리만 사용합니다.

## 바로 실행 가능한 것 (stdlib만, 모델/GPU/네트워크 불필요)

```bash
cd comparison
python3.12 -m asrbench preflight                       # 프로파일·lock·옵션 패키지 유무 확인 (로드/통신 없음)
python3.12 -m asrbench preflight --dataset my.jsonl    # 데이터셋 + WAV 형식 검증
python3.12 -m asrbench mock --out-dir results          # 합성 데이터 + mock 엔진 + FakeClock, 5개 run 생성
python3.12 -m unittest discover -s tests -v            # 테스트 (stdlib unittest)
python3.12 -m asrbench score results/<run_dir> [--dataset my.jsonl]   # 기존 결과 재채점
python3.12 -m asrbench server-command --profile nemotron-rc1-160ms --mode replay-native --binary <바이너리>  # 출력만
python3.12 -m asrbench postfilter-inspect              # ../ws_server.py를 ast로만 파싱해 추출 가능한 이름 나열
```

명시적으로 요청한 경우에만 외부를 조회합니다: `preflight --gpu`(읽기 전용 nvidia-smi),
`preflight --probe-server nemotron-rc1-160ms`(127.0.0.1:18101 의 `/version`, `/v1/models` GET).

## 설치 및 GPU 검증 완료

Kotoba v2 / Nemotron 모델은 공식 고정 revision으로 다운로드하고 SHA-256을 확인했습니다.
현재 small 모델도 기존 캐시에서 읽기 전용 복사했습니다. 세 프로파일에 로컬 모델 경로를 설정했습니다.
Kotoba GPU 실행은 기존 격리 Docker 이미지의 faster-whisper 1.2.1 / CTranslate2 4.8.0을 사용합니다.
호스트 Python에는 GPU 의존성을 전역 설치하지 않았으므로 일반 `asrbench run`의 faster-whisper 실행에는
해당 의존성 환경이 별도로 필요합니다. 검증된 `gpu_smoke.py`는 Docker를 직접 사용합니다.
NeMo-Speech는 공식 CUDA 12 릴리스 v0.1.0이며 `lock.json`은 실제 릴리스 커밋 `4f967622…`를 기록합니다.
원래 검토했던 소스 커밋은 `previously_reviewed_source_commit`에 별도로 보존했습니다.

최종 **33개 CPU 테스트 PASS**, 두 모델 모두 **일본어 GPU 전사 성공 및 메모리 반환 확인**.
실제 설치 상태, 시간 구분, 수동 전환 방법과 증거 경로는 [GPU_SETUP.md](GPU_SETUP.md)를 참조하세요.

```bash
python3 gpu_smoke.py --model kotoba --allow-gpu
python3 gpu_smoke.py --model nemotron --allow-gpu
```

위 두 명령은 기존 ASR이 정지된 테스트 시간에 하나씩 실행하며, 운영 중이면 거부합니다.
도구는 라이브 서비스를 자동 중지하지 않습니다. 이번 검증 후 기존 small ASR은 복구됐습니다.

## 데이터셋 (사용자 제공)

JSONL 1행 = 1발화. 예: `examples/dataset.example.jsonl`. 오디오는 **PCM16 / 16 kHz / mono / 비압축 WAV**.
필수 필드: `utt_id, session_id, order, audio_path, reference(null 허용), reference_verified,
speech_start_ms, speech_end_ms, nickname_candidates, spoken_nicknames(null 허용)`.

- `reference` / `spoken_nicknames` / `speech_*_ms` 는 **사람이 수동 작성한 정답**이며 채점에만 사용.
  엔진에는 `RecognitionInput`(오디오 + 바이어스 후보)만 전달되고 정답은 절대 전달되지 않음.
- `speech_start_ms` / `speech_end_ms`: WAV 샘플 0 기준의 수동 발화 시작/끝.
- Whisper 프롬프트(`prompt_previous_chars`)는 같은 세션의 **이전 인식 결과**로만 구성(정답 아님).

## 실제 실행 (명시적 opt-in, 수동 배타 윈도우에서만)

현재 장비는 **RTX 3070 Laptop 8GB** 이며, 현재 운영 중으로 확인된 GPU 서비스는 `npc_whisper`(ws_server)뿐입니다
(nemo-speech는 운영 중이 아님).
**절대 간섭하면 안 됩니다.** 하네스는 서비스를 자동으로 정지하지 않습니다. 운영자가 직접 정지/대기하여
배타 윈도우를 확보한 뒤 실행하세요. 다른 GPU 프로세스나 라이브 서비스로 보이는 프로세스가 있으면 차단됩니다.
WSL2에서는 nvidia-smi가 프로세스별 GPU 사용을 보여주지 못하므로, Windows 작업 관리자 등으로 직접 확인한
후 `--operator-attest-gpu-exclusive` 를 붙여야 하며 그 사실이 결과에 기록됩니다. 동시에 1개 모델만
(`BenchLock`).

```bash
# faster-whisper (in-process)
python3.12 -m asrbench run --profile whisper-small --mode replay-controlled --dataset my.jsonl \
  --allow-inference --operator-attest-gpu-exclusive [--bias on]
# Nemotron: server-command 출력을 운영자가 직접 기동 → attestation 작성 → 실행
python3.12 -m asrbench run --profile nemotron-rc1-160ms --mode replay-native --dataset my.jsonl \
  --allow-inference --operator-attest-gpu-exclusive --server-pid <PID> --attestation att.json
```

Nemotron은 모드마다 서버 설정이 다릅니다(controlled = 서버 endpointing 비활성, native = 활성).
`examples/attestation.example.json` 을 복사해 실제 값으로 작성하면 프로파일·lock과 대조됩니다.

## 모드와 비교 가능성

| 모드 | 내용 | 측정 |
|---|---|---|
| `file-controlled` | WAV 전체를 1회 디코드/요청 | `processing_ms` 만. 지연 필드는 **항상 null** |
| `replay-controlled` | 같은 오디오를 실시간 속도로 재생, **공통 호스트 에너지 엔드포인터**(profile `endpoint`)가 Whisper 디코드/Nemotron commit 시점을 결정 | 엔진 간 지연 비교는 이 모드 안에서만 유효 |
| `replay-native` | Nemotron 전용. 전체 재생, 서버 네이티브 endpointing(160ms/320ms 우측 컨텍스트, `stop_history_eou_ms`=800은 출발점일 뿐 튜닝값 아님) | 엔드포인터가 다르므로 controlled와 **직접 비교 불가** |

공통 엔드포인터(운영 동작 반영): 피크 ≥ 0.001 인 첫 프레임부터 버퍼링(0.02 미만의 조용한 선행 오디오 포함,
완전 무음은 버퍼를 열지 않음), 0.02 미만이 1500ms 지속되면 종료, 20s 강제 절단, **버퍼 길이** 400ms 미만은
필터링(별도 기록, 채점에 영향 없음). replay-controlled는 깨끗하게 끝난 세그먼트 1개만 채점하며, 세그먼트 없음 →
빈 가설(""), 다중/초과/종료 미달 → 실패로 기록. 프레임 100ms 단위 평가는 운영 클라이언트 청크 크기를 재현하지 않는
선언된 가정입니다.

## 지표

- **CER**: NFKC → casefold → 공백·구두점(P*) 제거 후 문자 편집거리 micro 평균. 가나↔한자·숫자 변환 없음.
  reference 가 null 이면 CER **null**(추정하지 않음), 실패는 별도 집계 + "실패=전부 삭제" 버전 병기.
- **닉네임**: 정규화 후 최장일치·비중첩 카운트를 수동 `spoken_nicknames`와 비교. `mention_recall`,
  `candidate_list_precision`. 오삽입은 후보 목록(디스트랙터 포함)에 있는 이름만 측정 가능.
- **지연**: `final_latency_ms` = 마지막 내용 있는 final 수신 시각 − (재생 원점 + 수동 speech_end_ms).
  음수면 `early_final`. `first_partial_latency_ms` = 첫 delta − (원점 + speech_start_ms), 있을 때만.
  final이 여러 개면 각각 기록(`finals`, `multiple_finals`), EOF commit의 빈 flush final은 보존하되 내용을 덮어쓰지 않음.
  백분위수는 nearest-rank.
- **VRAM**: nvidia-smi로 샘플링한 **디바이스 전체** memory.used (모델 단독 할당량 아님, 짧은 피크 누락 가능).

## 출력

`results/<시각>_<profile>_<mode>_bias-<on|off>[_MOCK]_<난수>/` — 매번 새 디렉터리, 기존 파일은 덮어쓰지 않음(`x` 모드).

- `events.jsonl`: 모든 원시 이벤트(수신 시각, 원점 기준 ms, 원본 payload)
- `utterances.jsonl`: 발화별 가설·원문·모든 final·지연·세그먼트·실패·정답 스냅샷(인식 후 첨부)
- `run.json`: 메타데이터(도구/파이썬 버전, 프로파일 전체+sha256, lock sha256·모델 revision, 엔진 describe,
  오디오/데이터셋 sha256, 클럭, GPU·간섭 판정·attestation, VRAM, mock 라벨) + 요약 + 발화별 점수
- `score-*.json`: `score` 명령의 재채점 결과

## 한계

- **mock**: 톤/무음 합성 오디오와 스크립트 출력. 파이프라인 검증용이며 모델 품질·속도에 대해 아무것도 말하지 않음.
  합성 reference는 전사가 아닌 placeholder.
- **후처리 패리티 갭**: 운영 블록리스트/반복 검출 필터는 프로파일 `postfilter.filters`에 나열하기 전까지 적용되지
  않으며 `parity: UNVERIFIED_GAP`으로 기록됨. `postfilter-inspect`로 순수 함수 이름을 확인 후 `extract_names`/
  `filters`에 추가(ast 추출 + 격리 네임스페이스, import 없음). 추가해도 운영 동작과의 일치는 수동 검증 필요.
- Nemotron 결과는 후처리 없는 raw 출력.
