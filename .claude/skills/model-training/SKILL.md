---
name: model-training
description: 호흡음 분류 모델 학습 표준 절차. LR+MLP 앙상블 학습, Youden index 임계값 최적화, 환자 단위 GroupShuffleSplit 분할(학습 중 누수 방지), 배포 아티팩트 4종 생성. 모델 학습/재학습, 성능 개선, 임계값 재산출/조정, 전이학습 실험, "모델 다시 학습" 요청 시 반드시 이 스킬을 사용할 것. 학습 없이 기존 산출물의 사후 검증만 원하면 pipeline-qa를 사용.
---

# Model Training — 호흡음 분류 모델 학습 표준 절차

로그멜 특성으로 배포 가능한 앙상블 모델을 학습하는 표준 프로토콜. 기준 스크립트: `Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/5 train_group_split_ensemble_thresholds.py`

## 학습 규약

| 항목 | 규약 | 이유 |
|------|------|------|
| 데이터 분할 | GroupShuffleSplit, 그룹=환자 ID, train:val=9:1 | 같은 환자의 세그먼트가 양쪽에 들어가면 성능이 과대평가됨 (환자 누수) |
| 전처리 | StandardScaler (train으로 fit, val은 transform만) | val로 fit하면 정보 누수 |
| 모델 | Logistic Regression + MLP | LR의 일반화 + MLP의 비선형 표현력 |
| 앙상블 | p = 0.5·LR + 0.5·MLP | 검증된 기준선 |
| 임계값 | 클래스별 ROC에서 Youden index(TPR−FPR) 최대점 | 클래스 불균형 보정 |
| 클래스 이름 | Healthy, Crackle, Rhonchi, Wheezing, Non-breathing | 학습 내부 순서는 알파벳순이어도 됨. 단 `thresholds.json`의 `class_names`가 확률 벡터 인덱스와 정렬되어 있어야 앱이 이름 기반으로 재정렬 가능 |

## 표준 절차

### Step 1: 분할 및 누수 검사
1. 파일명 접두어에서 환자 ID 추출 → GroupShuffleSplit
2. **분할 직후 반드시 실행**: `set(train_groups) & set(val_groups) == ∅` assert
3. 분할 결과(환자 수, 세그먼트 수)를 리포트에 기록

### Step 2: 학습
1. StandardScaler fit(train) → LR, MLP 학습
2. val 세트에서 앙상블 확률 산출

### Step 3: 임계값 최적화
- 클래스별 one-vs-rest ROC → Youden index 최대 임계값
- 샘플이 부족한 클래스는 기본값 0.5 사용, 리포트에 명시

### Step 4: 아티팩트 저장
`_workspace/02_train_artifacts/`에 4종 완전 저장:
- `scaler.pkl`, `model_lr.pkl`, `model_mlp.pkl`, `thresholds.json`
- `thresholds.json`에는 `class_names`를 포함하고, 그 순서가 모델 확률 벡터의 인덱스 순서와 일치해야 한다 (앱이 이 이름 목록으로 UI_ORDER에 재정렬함)

### Step 5: 평가 리포트
`_workspace/02_train_report.md`에 기록:
- 클래스별 precision / recall / F1 + confusion matrix (accuracy 단독 보고 금지 — 클래스 불균형이 큼)
- 이전 실험/기준선 대비 비교표
- 입력 특성의 파라미터 (data-engineer 리포트에서 가져와 명기 — QA 교차 검증용)

## 고급 실험 시

step17A~25A, OPERA-CT 실험(Development/ 하위) 계보를 이을 때:
- **평가 프로토콜(분할 방식·지표)은 바꾸지 않는다.** 프로토콜이 다르면 이전 실험과 비교 불가능해져 실험 자체가 무의미해진다.
- 새 실험은 `Development/` 하위에 `stepXX{A}_{설명}` 형식의 새 폴더로 만들고, 기존 실험 폴더를 수정하지 않는다.
- 전이학습(OPERA-CT)은 `Development/TRANSFER_LEARNING_ROADMAP.md`의 계획 단계(Fine-tuning, Custom Heads, Hybrid)를 참조하고, 완료 후 로드맵에 결과를 갱신한다.

## 주의사항

- 재학습 시 이전 아티팩트를 덮어쓰지 않는다. 새 `_workspace/02_train_artifacts/`에 만들고, 배포 반영은 deployment-engineer가 버전 폴더로 수행한다.
- 학습 데이터가 바뀌었는데 성능이 크게 뛰면 먼저 누수를 의심한다. 좋은 소식일수록 검증한다.
