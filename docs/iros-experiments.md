# 인터넷등기소 독립 비교 실험

EXP-001~005 실행 절차다. 기존 train 1,197장은 수동 검수 완료지만 신규 평가 데이터가 없어 실제 정확도 개선은 아직 판정하지 않았다. 모든 명령은 저장소 루트에서 실행한다. 새 패키지 설치는 필요 없다.

## 데이터 준비

신규 이미지 500장 이상을 모델 예측·신뢰도와 관계없이 수집 순으로 확보해 수동 검수한다. `<6자리 정답>.png`로 별도 폴더에 저장한다. 파일 mtime은 수집 순서를 유지해야 한다(기존 웹 라벨 변경은 mtime 유지). 수집 순서를 복원할 수 없는 파일을 임의로 정렬해 대표 표본으로 간주하지 않는다. 같은 정답 파일명이 충돌하면 하위 폴더로 나누어 서로 다른 원본을 보존한다. 신규 경로는 하위 폴더까지 읽고 mtime 동률은 상대 경로로 정렬한다.

```bash
apps/web/.venv/bin/python -m web.experiments prepare \
  --study /tmp/iros-study-42 \
  --data-root captcha_data \
  --new-dir /absolute/path/reviewed-new \
  --reviewed --batch-size 64 --budget-minutes 240
```

`--reviewed`는 수동 검수와 예측 비선별 수집이 끝났다는 확인이다. 디렉터리의 존재나 숫자 파일명만으로 검수를 추정하지 않는다. 픽셀 중복을 기존 train 및 신규 표본 사이에서 제거하고, 남은 표본 중 수집 순 첫 500장을 seed 42로 검증 200장·최종 평가 300장으로 나눈다. 동일 픽셀에 다른 라벨이 있으면 실패한다. 기존 train 내 동일 픽셀/동일 라벨은 유지한다. 이미지 파일은 이동하지 않는다. 추가 신규 데이터는 다음 독립 실험용으로 남긴다.

study에는 manifest, 고정 프리셋, 운영 모델·sidecar의 복사본과 해시가 저장된다. 실행 때와 종료 때 전체 manifest를 검증하고, loader는 이미지마다 파일 해시를 재확인한다. 고정 파일·코드가 바뀌면 새 study가 필요하다. 작업 중 기존 이미지나 라벨을 수정하지 않는다.

## 비교 실행

```bash
apps/web/.venv/bin/python -m web.experiments run --study /tmp/iros-study-42 --suite
```

전체 실행은 기존 모델의 검증셋 기준 측정 → A~E(seed 42) → F → 상위 B~E 2개의 seed 17·2026 순이다. F는 B~E의 seed 42 최상 증강 설정을 사용한 scratch 대조군이다. 다음처럼 한 실행씩 진행할 수도 있다.

```bash
apps/web/.venv/bin/python -m web.experiments run --study /tmp/iros-study-42 --preset B --seed 42
apps/web/.venv/bin/python -m web.experiments evaluate --study /tmp/iros-study-42 --run baseline
apps/web/.venv/bin/python -m web.experiments report --study /tmp/iros-study-42
```

| 프리셋 | 초기화 | 이미지 증강 | SpecAugment | LR | 최대/최소 에폭 | warmup/patience |
|---|---|---|---|---|---|---|
| A | scratch | current | current | 1e-3 | 80/40 | 0/15 |
| B | finetune | current | current | 1e-4 | 40/10 | 3/10 |
| C | finetune | weak | current | 1e-4 | 40/10 | 3/10 |
| D | finetune | weak | off | 1e-4 | 40/10 | 3/10 |
| E | finetune | weak | weak | 1e-4 | 40/10 | 3/10 |
| F | scratch | B~E 최상 | B~E 최상 | 1e-3 | 80/40 | 0/15 |

최소 에폭까지 patience를 누적하지 않고 이후부터 누적한다. 공통값은 batch 64, dropout 0.1, AdamW, weight decay 1e-4, gradient clip 5, Focal gamma 2/alpha 0.25이다. CNN/BiLSTM 구조·전처리는 유지한다. 미세조정은 전체 레이어 가중치를 로드하고 optimizer는 새로 만들므로 중단 재개와 다르다.

- weak 이미지 증강: 회전 ±3°, 이동 ±2%, 배율 0.98~1.02만 적용한다.
- current SpecAugment: time 최대 15×2, feature 최대 8×2. weak: 각 최대 4×1. off: 마스킹 없음.
- 검증·최종 평가는 증강 없이 FP32·beam width 10. 주 지표는 6자리 완전 일치율, CER은 Levenshtein 거리/정답 문자 수다. 반복 숫자는 인접 여부와 무관하게 같은 숫자가 두 번 이상 있는 라벨이다.
- AMP 초기 배율 때문에 gradient overflow가 발생하면 GradScaler가 해당 step을 건너뛰고 배율을 낮춘다. `numerical_warning` 이벤트에 에폭·배치·변경 전후 배율을 기록하고 에폭마다 실제 optimizer step/생략 횟수를 저장한다. loss NaN/Inf는 즉시 실패하며 전체 학습에서 유효 step이 0이면 후보를 확정하지 않는다.
- 매 에폭 validation 예측을 보존한다. `evaluate --run ...`은 검증 오답 원본·전처리 PNG를 `validation/<run>/review`에 저장한다. `3↔8` 혼동 등을 사람이 재검수하며 예측으로 정답을 자동 수정하지 않는다. 라벨 오류가 확인되면 해당 비교는 무효이며 수정 후 새 study로 시작한다.

시간 배분 목표는 기준/수치 30분, A~F 100분, 반복 70분, 최종 평가/export 40분이다. CLI는 개별 단계의 고정 배분 대신 누적 240분을 검사하고 suite 학습에서 마지막 40분을 유보한다. 시간 검사는 배치 경계에서 이루어지므로 진행 중인 단일 GPU 연산/디코딩/export 호출의 종료까지는 걸릴 수 있다. 실패 시간도 차감한다. 실패·시간 초과 실행의 임시 체크포인트는 후보에 포함하지 않는다.

OOM이면 실패 이력을 보존하고 **모든 비교를 batch 32인 새 study에서** 수행한다. 새 study의 `--budget-minutes`에는 이미 사용한 시간을 뺀 잔여 예산을 지정한다. 서로 다른 batch·장치·라이브러리 환경의 실행을 섞지 않는다. 완료된 run 디렉터리는 재사용하지 않으며 `.busy`는 동시 실행을 막는다. 강제 종료로 잠금이 남으면 프로세스 종료를 확인한 뒤에만 잠금을 정리한다. 미완료 실행을 완료로 편집하거나 실패를 지워 시간 예산을 복구하지 않는다.

## 후보 확정과 최종 평가

```bash
apps/web/.venv/bin/python -m web.experiments evaluate --study /tmp/iros-study-42 --final
apps/web/.venv/bin/python -m web.experiments report --study /tmp/iros-study-42
```

A~F 및 상위 2개 설정의 3 seed 결과가 모두 있어야 실행된다. 설정 순위는 3회 검증 정확도 평균, CER 평균, 총 실행시간 순이다. 우승 설정 내 체크포인트는 검증 정확도→CER→loss→시간 순으로 고정한다. 테스트를 읽기 전에 `selection.json`에 설정·체크포인트 해시를 기록하고 study의 학습·검증 재평가를 잠근다. 최종 실패 후에도 테스트를 보고 재튜닝할 수 없도록 잠금이 유지된다.

후보만 `candidate/`로 PT2/ONNX/ORT/sidecar를 export하고 기존 검증 도구로 PyTorch와 ONNX/ORT의 학습 이미지 8장 예측 일치·로짓 오차를 확인한다. 그 다음 후보와 기존 모델에 같은 최종 평가 300장을 한 번씩 적용한다. paired bootstrap 10,000회(seed 42)로 정확도 차이의 95% CI를 계산한다. 다음 조건을 모두 만족해야 보고서의 `deployable_candidate`가 참이다.

1. 필수 비교 전체 완료
2. 기존 대비 오답 20% 이상 감소(기존 오답이 0이면 통과 불가)
3. 정확도 차이 CI 하한 > 0
4. CER 악화 없음
5. export 동등성 검사 통과

불확실하면 기존 모델을 유지하고 추가 독립 평가 데이터를 확보한다. 운영 모델 교체는 이 도구가 수행하지 않는다. 같은 최종 평가셋에 맞춰 설정을 다시 조정하지 않는다.

## 검증과 근거

```bash
apps/web/.venv/bin/python -m pytest tests/test_experiments.py tests/test_experiment_cli.py -q
apps/web/.venv/bin/python -m pytest tests -q
```

CUDA CTC backward는 결정적 구현이 없어 경고 허용 모드로 실행한다. seed·분할 고정은 비트 단위 재현 보장이 아니다. `environment.json`에 이 제한과 PyTorch/CUDA/cuDNN·GPU·정밀도를 기록한다. [PyTorch 미세조정 문서](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)의 가중치 재사용과 새 optimizer 구성을 따른다.
