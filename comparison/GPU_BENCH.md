# 공개 샘플 GPU 속도 벤치 (`public_speed.py`)

whisper-small / kotoba-whisper-v2 (faster-whisper) 와 Nemotron 3.5 Q8 (nemo-speech 네이티브 서버) 의
**속도만** 비교한다. 정확도·CER·replay/발화 종료 지연은 측정하지 않으며, 전사 결과가 서로 다른 것은 예상된 일이다.

## 명령

```bash
python3 public_speed.py prepare                       # local/audio/public_speed/ 생성 (덮어쓰기 거부)
python3 public_speed.py run --allow-gpu --repeats 8   # 옵션: --nemotron-context3
python3 -m unittest tests.test_public_speed           # GPU 불필요
```

`run` 은 서비스를 정지·재시작하지 않는다. npc_whisper 정지/복원은 외부 래퍼(Codex) 담당이며,
npc_whisper 나 ws_server/nemo-speech 프로세스가 떠 있거나 18101 포트가 사용 중이면 실행을 거부한다.

## 샘플

`local/audio/sample_ja_speech.wav` (sha256 은 `sample_ja_speech.source.json` 과 대조) 에서 16 kHz PCM16 mono 로
6개 구간을 잘라낸다: short_a 10s+2.5, short_b 30s+3, medium_a 50s+6, medium_b 80s+8, long_a 110s+12, long_b 160s+18.
`samples.json` 에 원본 해시/리비전/URL/README 위치, 각 WAV 의 sha256/길이/오프셋을 기록한다. 정답 텍스트는 없다.

**라이선스 주의:** 이 음성의 출처·라이선스는 확인되지 않았다(호스팅 저장소의 모델 라이선스가 음성까지 포함한다는 보장 없음).
잘라낸 클립과 결과의 원문 텍스트는 재배포하지 말 것.

## 방법

- BenchLock 으로 단독 실행, 엔진은 한 번에 하나, GPU 0 만 사용. 락 fd 는 자식(docker/서버)에 상속되지 않으며 worker 는 락을 잡지 않는다.
- 엔진마다: 콜드(import / 모델 init, Nemotron 은 프로세스 시작→`/ready`) 별도 기록 → 최장 클립 warmup 2회(통계 제외)
  → 6클립 × 8라운드 = 48회, 라운드마다 한 칸씩 회전한 결정적 순서.
- Whisper: 기존 이미지 `whisper_streaming-whisper:latest` 를 `--gpus device=0 --network none` 으로 실행, ROOT 는 같은 절대경로 ro,
  결과 디렉터리만 rw. cuda/float16/local_files_only, language=ja beam=5 condition_on_previous_text=False VAD(min_silence 300 ms)
  temperature 0–1.0, initial_prompt 없음, no_speech 0.6 / logprob -1.0 (upstream 기본값). 측정 구간 = WAV 디코드+VAD+인식(제너레이터 소비까지).
  프로덕션 후처리 필터는 적용하지 않은 raw ASR 텍스트.
- Nemotron: 설치된 바이너리를 `127.0.0.1:18101`, `--asr.backend.gpu 0 --no-warmup --asr.batching.enabled=false
  --asr.endpointing.enable=false --asr.streaming.rnnt_right_context 1` 로 직접 띄움. `/v1/models` 모델 ID, `/ready` CUDA 를 확인.
  `/v1/audio/transcriptions` (ja-JP, verbatim=true, json). 측정 구간 = 파일 읽기+multipart 생성+HTTP+JSON 수신이므로
  **API 오버헤드가 포함**되며, Python API 를 직접 부르는 Whisper 측과 조건이 다르다(순수 커널 시간 아님).
- `--nemotron-context3`: rc1 서버가 완전히 종료·회수되고 VRAM 이 해제된 뒤에만 rc=3 으로 재측정. 이는 *네이티브 파일 모드 context* 이며 스트리밍 replay 가 아니다.
- VRAM: `guards.VramSampler` 로 GPU 전체 memory.used 의 baseline/peak/delta (프로세스별 값 아님). 엔진 종료 후 baseline+100 MiB 이하로
  30초 내 돌아오지 않으면 중단.
- 타임아웃: worker/서버 기동 300 s, HTTP 요청 30 s, 전체 20분. 소유한 프로세스/컨테이너만 finally 에서 정리하고 회수·삭제를 확인. 서비스 자동 kill 없음.

## 결과

`results/public_speed/<JST시각>_<uuid>/` (기존 디렉터리 덮어쓰기 없음): `run.json`(실제 CLI/설정/버전/GPU/VRAM),
`<engine>.trials.jsonl`(호출마다 elapsed_ms, RTF, duration, raw_text, error — 즉시 fsync), `summary.json`,
컨테이너/서버 stdout·stderr 로그. 요약은 nearest-rank p50/p95: 모델별 pooled(n=48) 와 클립별(n=8, 분포 참고용).
에러와 빈 텍스트는 보존하되 성공으로 세지 않는다. 로드 포함 비교는 하지 않는다.

## 2026-10-01 실제 GPU 속도 결과

결과: `results/public_speed/20261001T162945_265089cd/`. **16:31 JST baseline small 복구 완료**.

공개 일본어 강연 1개의 서로 다른 6구간. 각 모델/구성은 18초로 2회 워밍업 후 구간별 8회, 총 48회 측정.
세 모델 및 Nemotron 추가 반복 모두 CUDA 확인, warm 192회 + warmup 8회 오류 0. 정확도 정답은 없어 CER 미산출.

| 입력 길이 / 시작 위치 | small p50(ms) | Kotoba p50(ms) | Nemotron 파일 p50(ms) |
|---|---:|---:|---:|
| 2.5s / 10s | 375 | 344 | 26 |
| 3s / 30s | 431 | 341 | 30 |
| 6s / 50s | 629 | 514 | 49 |
| 8s / 80s | 875 | 571 | 75 |
| 12s / 110s | 1116 | 691 | 124 |
| 18s / 160s | 1802 | 1245 | 158 |

| 모델 | pooled p50 / p95(ms), n=48 | RTF p50 | GPU 전체 baseline / peak / delta(MiB) |
|---|---:|---:|---:|
| small | 731 / 1909 | 0.1087 | 865 / 1755 / 890 |
| Kotoba | 538 / 1268 | 0.0764 | 865 / 2867 / 2002 |
| Nemotron offline | 59 / 165 | 0.0096 | 857 / 2033 / 1176 |

처리시간은 Nemotron < Kotoba < small 순서였습니다. 같은 구간 p50 기준 Kotoba는 small보다 약 1.09–1.62배, Nemotron은 약 9.0–14.6배 빨랐습니다.
Whisper는 파일 디코드/VAD/인식 직접 호출, Nemotron은 로컬 HTTP 비용까지 포함합니다. 동일 파일→완성 텍스트 작업의 실용 측정이며 순수 GPU 커널 비교가 아닙니다.

**중요한 검증 결과:** rc1/rc3라는 파일명은 요청한 streaming 설정을 나타낼 뿐입니다. 실제 양쪽 서버 로그는 모두 `mode=offline ... attention-right=3`입니다.
따라서 rc3 결과는 동일 offline 경로의 두 번째 반복으로만 해석합니다. **160/320ms 스트리밍 또는 실제 음성 재생 실험을 했다고 주장하지 않습니다.**

cold는 워밍업/인식에서 분리했습니다: small import 1.616s + 모델 초기화 0.982s, Kotoba import 1.565s + 모델 초기화 3.448s.
Nemotron 프로세스 시작→HTTP 준비 0.405s; 첫 18초 요청 0.569s, 두 번째 워밍업 0.153s. 모델 로드 경계가 달라 초기화 숫자로 엔진 속도 순위를 매기지 않습니다.
OS 파일 캐시를 비우지 않았습니다. 100ms 간격 GPU 전체 사용량 샘플이며 프로세스별 할당량은 아닙니다.
각 단계 종료 후 프로세스/컨테이너 해제 및 메모리 baseline+100MiB 이내 반환 확인. WSL 프로세스별 GPU 회계 한계가 있으며 별도 Windows VRChat/python/ollama 후보 프로세스는 사전 검사에서 관찰되지 않았습니다.

현재 실사용 small은 발화 후 무음 1.5초를 기다립니다. 이 대기는 표의 처리시간에 들어 있지 않습니다. 단순히 처리시간만 줄어드는 만큼 NPC가 즉시 응답한다고 해석하면 안 됩니다.
실제 발화 종료→최종 인식/LLM/TTS 지연, nickname 정확도, native endpoint 정책은 이번 범위 밖입니다. Kotoba의 일부 전사 생략과 Nemotron의 일부 오인식/반복도 원문에 보이므로 속도만으로 교체하지 마세요.
p95는 nearest-rank. 구간당 n=8의 p95는 최대값이며 신뢰할 만한 장기 꼬리 지연 추정이 아닙니다. pooled 수치는 길이 혼합이라 구간별 표를 우선 보세요.

원문/개별 trial/설정/버전/해시/샘플 출처와 라이선스 확인 상태/서버 로그는 결과 디렉터리에 보존했습니다. 원래 raw 파일은 변경하지 않고 `analysis.json`에 해석을 추가했습니다.
최종 CPU 테스트 36개 PASS. 정상 서비스 복구 증거는 `results/public_speed/restoration-20261001T072932Z.json`.
