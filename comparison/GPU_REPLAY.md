# GPU replay（public clips, 遅延のみ）

`public_replay.py` は公開サンプル 3 クリップ（`short_a` 2.5 s / `medium_b` 8 s / `long_b` 18 s,
`local/audio/public_speed`）を実時間ペースで 20 ms / 16 kHz mono PCM パケットとして再生し、final の到着遅延を測る。
**精度（CER）は評価しない。** 本番サービスの停止・再起動（baseline 管理）は行わない。外側のラッパーが停止し、
finally で復旧させる。

```
python3 public_replay.py run --allow-gpu     # 予算 20 分, 実測 ~10 分想定
python3 -m unittest tests.test_public_replay  # GPU 不要
```

## 構成（逐次実行, 同時に動かす ASR モデルは常に 1 つ, GPU 0）

| config | エンジン | エンドポイント |
|---|---|---|
| `small` / `kotoba` | 既存 docker image 内の `FasterWhisperEngine`（python3.11, `--network none`, ROOT ro + 出力 rw） | host 共通 1500 ms（controlled） |
| `nemotron_ctl_rc1` | native server rc1, `--asr.endpointing.enable=false` | host 共通 1500 ms → commit |
| `nemotron_nat_rc1` / `_rc3` | native server, endpointing ON, `vad_based=false`, `stop_history_eou_ms 800` | server native, EOF で commit |

各 config: short warmup 1 回（統計から除外）+ 3 クリップ × 3 回 = 計測 9 試行。コールド起動（model load /
server ready）も統計から分離。utterance ごとに新規セッション、prompt 引き継ぎなし。config 間で VRAM の返却を
確認してから次を起動。Whisper の postfilter は profile の filters が空のため何も import されない（`not_configured`）。

## 指標 `clip_end_to_final_ms`

- アンカー = **制御されたクリップ終端（人工的な急な切断, 最終 100 ms の peak > 0.02）+ 付加 3000 ms のゼロ**。
  手動アノテーションした自然な文末ではない。自然発話のポーズでの end-of-speech 遅延を主張する根拠にはならない。
- decode/処理時間を含む。
- controlled: commit 後の最後の非空 final。host endpointer の segment が voice_end = クリップ長, endpoint = +1500 ms
  であることを毎試行検証。
- native: commit **前**の最後の非空自動 final。無ければ `no_native_final`（EOF commit の final で代用しない）。
  commit 後の final（空の EOF final 含む）は別に保存。負値は `early_final`（成功扱いしない）。複数の自動 final は
  個別に記録し、テキストは連結しても遅延は最後の final で評価する。
- エンジン比較として扱えるのは controlled 表のみ。native 800 は別の endpointer なので別表（`native_800`）。
- オフライン速度（`public_speed.py`）から native のスループット順位は導かない。

## 出力 `results/public_replay/<JST時刻>_<uuid8>/`（上書きしない）

`run.json`（argv, profile/clip/実装 sha256, server の `[asr]` 実効ログ, VRAM, cold init）, `summary.json`
（n / p50 / p95 / max, max send lag, status 内訳, errors）, `<config>.trials.jsonl`, `<config>.events.jsonl`
（全イベント, `t_rel_ms` = replay origin 基準）, server/worker の stdout/stderr ログ。

いずれかの config で計測 9 件未満、error、backend 不一致（`backend=CUDA0` / `mode=streaming` / `right=N` が
ログに無い、`/v1/models` が `cuda:0` でない）なら exit 1。

## 2026-10-01 실제 재생 검증 완료

결과 `results/public_replay/20261001T164908_5b56d228/`. 16:58:59 JST small 모델 로드 완료와 8100/asr 서버 재시작까지 확인.
16kHz PCM16을 20ms씩 실시간 속도로 전송. 2.5/8/18초 공개 클립 각각 3회, 구성별 warmup1회 제외. 측정45회 모두 유효, CPU 테스트42개 PASS.
**시간 기준은 자연 발화의 수동 문말이 아니라, 말소리가 있는 공개 클립을 인위적으로 자르고 무음3초를 붙인 정확한 경계입니다.**

기존 파일 실험이 offline이었던 원인: HTTP 업로드 경로는 `recognizer->recognize`(공식 릴리스 server/http/http_server.cpp:597), WS 경로는 `streaming_recognize`(:885)를 호출합니다.
공식 소스: https://github.com/NVIDIA/NeMo-Speech.cpp/blob/4f9676226f667d14608487df744f375db87127f8/server/http/http_server.cpp

### 공통 host 무음 1500ms 조건 (클립 경계→내용 있는 최종 결과, 중앙값 초)

| 음성 길이 | small | Kotoba | Nemotron streaming RC1 |
|---|---:|---:|---:|
| 2.5s | 2.010 | 2.069 | 1.623 |
| 8s | 2.482 | 2.163 | 1.627 |
| 18s | 3.332 | 2.574 | 1.640 |

### Nemotron native token-silence 800ms (위 표와 종단 정책이 다름)

| 설정 | p50 / p95 초(n=9) | 실제 로그 |
|---|---:|---|
| RC1 | 1.300 / 1.380 | streaming right=1, step=160ms, CUDA0 |
| RC3 | 1.540 / 1.720 | streaming right=3, step=320ms, CUDA0 |

최대 전송 스케줄 지연 5.04ms. 모든 native final은 EOF commit 전에 발생했고, 빈 flush final은 최종 지연으로 대체하지 않았습니다.
실제 로그에서 `backend=CUDA0`, `mode=streaming`, `right=1 step=160ms` 및 `right=3 step=320ms`를 모두 확인했습니다.
Whisper 계열은 동일 host 종단 판정 뒤 decode하고 Nemotron controlled는 같은 시점에 commit합니다. 비교는 이 controlled 표 안에서만 합니다.
Native 800ms는 token-silence 방식으로 에너지 기준1500ms와 다르며 실제 확정 시간이800ms라는 보장은 아닙니다. 숫자는 로컬 시험 클라이언트의 WS 전달 비용을 포함합니다.
구간당3회(n=3), 혼합9회여서 p95는 최대 관측치에 불과합니다. 미검증 자연 문말, VRC/Gateway/LLM/TTS 경로, 정확도/CER는 포함하지 않습니다.
운영 Whisper의 blocklist/반복 후처리는 적용하지 않은 명시적 parity gap이 있습니다. 공통 종단 정책을 재현한 하네스이며 운영 전체 경로 측정은 아닙니다.
한 번에 모델 하나만 실행, 각 단계 VRAM 반환 확인, 실험 컨테이너/서버 없음. 기존 offline 결과는 보존했습니다.
복구 증거: `results/public_replay/restoration-20261001T074855Z.json`.

**Native 종단 정책 관찰:** RC1/RC3 각각 9회 중 3회에서 클립 종료 전에 내용 있는 중간 final이 추가로 발생했습니다. 마지막 final은 모두 클립 종료 이후였고 EOF commit에 남은 내용은 없었습니다. 이 내부 분할을 자연 대화에서 너무 일찍 응답하는지 검증해야 하므로 800ms를 그대로 운영 권장값으로 확정하지 않습니다. 상세 시각은 `endpoint-observations.json`에 보존했습니다.
