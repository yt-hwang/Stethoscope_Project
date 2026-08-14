# ML 기술 검토 노트 — 리뷰 대응 준비 자료

**논문:** "A Soft Wearable System for Real-time Respiratory Disease Diagnosis and Vibrotactile Intervention" (Science Advances 투고)
**작성일:** 2026-08-13
**범위:** ML 파이프라인(분류 모델·실시간 추론) 관련 기술 사항. 하드웨어/촉각 액추에이터/임상 프로토콜은 담당 표시만 하고 상세 제외.
**근거 코드:** `Paper/Final_Code_Manuscript/` (원본: `Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/` + `6 Final Deployment to Dr.Oh/`)
**감사 전문:** `_workspace/04_qa_paper_code_audit.md` (검증 스크립트: `_workspace/qa_scripts/01~03`)

---

## 1. 논문에서 ML 관련 부분 (파일 · 페이지 · 라인)

### 1.1 Manuscript_SA.pdf (25쪽)

| 페이지 | 라인 | 내용 | 비고 |
|--------|------|------|------|
| **p.7–8** | **L342–360** | **ML 파이프라인 명세** — 구간 파싱 → breathing/non-breathing 변환 → 16 kHz 리샘플 → 세그먼트 5-라벨 → 2 s 윈도우/0.5 s 홉(+padding 주장) → 64-band log-Mel per-sample 정규화 → GroupShuffleSplit(train/val/test 주장) → LR+MLP 앙상블 → 38명 학습 주장 | 우리 코드의 직접 서술. §4의 불일치 다수가 이 구간에 있음 |
| **p.8** | L361–381 | 실시간 추론 — 앙상블 설계 근거(선형+비선형), Fig 4B 성능(≥85%), Fig 4C 실시간 트레이스, breathing/non-breathing을 5-클래스 출력의 그룹핑으로 도출 | L374–376의 플랫 그룹핑 서술이 코드와 일치 |
| p.8–9 | L382–400 | 알림·액추에이션 프로토콜 — 비정상 클래스 또는 HR/BR 임계 초과 시 알림, 확신도 >70% 사례 | 앱 미구현 항목 포함 (§4.8, §4.9) |
| p.3–4 | L153–168 | Fig 1E 시나리오 — "2 s 윈도우, **0.5 s마다** boundary shifting 업데이트", "먼저 breathing/non-breathing 분류 → breathing만 4-클래스" (계층 서술) | 두 서술 모두 코드와 불일치 (§4.5, §4.7) |
| p.11 | L501–527 | Human subject study — 43명 등록(5 healthy + 32 성인 + 8 소아), 폐과 전문의 라벨 = ground truth, "38 patients used for model training"(L357) | 실 사용량과 상이 (§4.3) |
| p.10 | L487–492 | Discussion — 고정 2 s 윈도우 한계 인정, adaptive segmentation·pretrained audio model(AudioSet 계열) 전이학습 향후 방향 | R2#9 답변 소재 + 우리 OPERA-CT 실험(Development/)과 연결 |
| p.24 | Fig 4 | A: 파이프라인 도식(19 세그먼트→38 윈도우, linear/non-linear 분기, softmax fusion), **B: 5-클래스 confusion matrix** (대각: Healthy 89 / Wheezing 85 / Crackle 87 / Rhonchi 94 / Non-breathing 65 %), C: 30 s 실시간 추론, D: 알림 워크플로(`Ths > x`), E–G: 클래스별 사례 | **B의 산출 근거가 로컬에 없음** (§4.1) |

### 1.2 SI_SA.pdf (37쪽)

| 페이지 | 항목 | 내용 | 비고 |
|--------|------|------|------|
| **p.29–30** | **Table S2** | "모델 학습에 사용된 38명" 상세 — #1–32 성인(P/A, 폐엽 위치, Wheezing/Crackle/Rhonchi, 진단), #33–38 소아(A-RUL, Wheezing) | 학습 데이터 명세의 근거표. 실제 특성 추출 사용분과 대조 필요 (§4.3) |
| p.28 | Table S1 | 코호트 인구통계·모집 기준·sound type | R2#4 대응 근거 |
| p.31–32 | Table S3 | 비교표 — 본 연구 "Multiclass Classification (>85%)", "Real-time model deployment" | Fig 4B 주장의 반복 |
| p.16–17 | fig S9–S10 | 임상 프로토콜 (cross-sectional 37명 / longitudinal 8명, CRS) | |
| p.23 | fig S16 | 수집 UI + 클라우드 저장 | 학습 데이터 수급 경로 |

### 1.3 리뷰 리포트 (4쪽) — ML 관련 코멘트 위치

- p.1: R1 #1(과장 주장), **#4(윈도우 단위 알림 → 집계 필요)**
- p.2: R2 #1(소리≠질병), #3(라벨링 절차), **#5(세트별 참가자·녹음 수)**, **#6(클래스별 지표)**
- p.3: R2 **#8(레이턴시)**, **#9(고정 윈도우·패딩)**, R3 **Major #2(non-breathing 65% 혼동, FP/FN 알림율)**
- p.4: R3 Minor #3(end-to-end 레이턴시)

---

## 2. 구현된 ML 파이프라인 기술 명세 (코드 검증 기준)

논문 서술이 아니라 **코드가 실제로 하는 일**. 리뷰 답변은 이 명세를 기준으로 작성해야 함.

### 2.1 데이터 → 특성

```
[Excel 라벨] ML test sound list breathing info.xlsx
  행1: 파일명 | Inhale | Exhale | ... , 행2: 진단 | t_start | t_end | ...
    │  1 parse_breathing_intervals.py  (DURATION_SEC=30.0)
    ▼
[JSON] breathing: [[t0,t1],...] / non_breathing: 여집합
    │  2 extract_segments_from_json.py  (TARGET_SR=16000)
    ▼
[구간 세그먼트] breathing → 진단 클래스(Healthy/Wheezing/Crackle/Rhonchi), 여집합 → Non-breathing
    │  3 make_segments_2s_hop050.py  (WIN=2.0 s, HOP=0.5 s)
    │    - 슬라이딩 윈도우, 라벨은 coverage 기반(choose_label_by_coverage)
    │    - 2 s 미만 꼬리는 폐기 (if i1 > n: break) — 패딩 없음
    ▼
[윈도우 wav] Augmented_Windows_by_Coverage/{클래스}_window/  (N=1,368)
    │  4 extract_Logmel.py
    │    - librosa melspectrogram: SR 16000, n_mels 64, fmin 50, fmax 7900,
    │      win 64 ms, hop 32 ms → power_to_db → 시간축 평균 → 64-dim 벡터
    │    - per-sample 표준화: (v − μ_v) / σ_v
    ▼
[features_64mel.npz] X: (1368, 64), y, groups
```

- **그룹 키**: 파일명에서 시간범위 접미어를 제거한 stem (녹음/세션 단위, `_1/_2` 세션 접미어 포함). 환자 단위 축약 시 21명.
- **클래스 분포 (N=1,368)**: Non-breathing 718 (52.5%) / Wheezing 289 / Crackle 149 / Rhonchi 121 / Healthy 91 — **심한 불균형**.
- **RAW 입력의 실효 대역**: 원본 wav native 4 kHz (BLE 수신 그대로) → 16 kHz 업샘플. 나이퀴스트 2 kHz 초과 정보 없음 → **64 멜밴드 중 상위 ~29개(≈45%, fmax 7900 설정분)는 무정보**. 학습·추론 동일 조건이므로 정합성은 유지되나 논문 주파수 서술과 조율 필요.

### 2.2 학습 (`5 train_group_split_ensemble_thresholds.py`)

| 항목 | 구현 |
|------|------|
| 분할 | `GroupShuffleSplit(n_splits=1, test_size=0.10)` — **train/val 9:1 단 1회, test 세트 없음**. 결과: train 21그룹/1,197윈도우, val 3그룹/171윈도우, 그룹 겹침 0 |
| 전처리 | `StandardScaler` (train fit → val transform) |
| 모델 | `LogisticRegression` + `MLPClassifier` |
| 앙상블 | `p = 0.5·p_LR + 0.5·p_MLP` (고정 동일가중) |
| 임계값 | 클래스별 OVR `roc_curve` → Youden J(TPR−FPR) argmax → `thresholds.json` |
| 클래스 순서 | canonical 알파벳순 `[Crackle, Healthy, Non-breathing, Rhonchi, Wheezing]` — thresholds.json `class_names`가 확률 인덱스와 정렬 |
| 산출물 | `scaler.pkl, model_lr.pkl, model_mlp.pkl, thresholds.json` + `summary.json`(카운트만, **성능 지표 저장 안 함**) |

### 2.3 실시간 앱 (`Stethoscope_App_YH_V1.py`)

```
BLE (UUID 0000eef2-…, 4,000 Hz, 180 B 패킷)
 → 16 kHz 업샘플 → 2.0 s 버퍼 → HOP_S = 1.0 s 마다 추론   ← 논문은 0.5 s 주장
 → 동일 log-Mel 64-dim (파라미터 학습과 완전 일치, 검증됨)
 → scaler → 0.5·LR + 0.5·MLP → 5-클래스 확률 (플랫, 계층 아님)
 → UI: thresholds.json class_names로 UI_ORDER(Healthy, Crackle, Rhonchi→"Bronchi", Wheezing, Non-breathing) 재정렬
 → 확률 테이블/그래프 표시.  argmax(p − τ) 진단은 계산 후 폐기(_=), 알림 트리거 미구현
```

- `tetho_softmax_export.py`: 동일 파라미터의 오프라인 배치 추론(단, HOP 0.5 s)→ CSV/그래프. Fig 4C류 트레이스 생성 용도.
- 앱 3개 사본(`4 Realtime Pipeline_New`, `6 Final Deployment`, 논문 사본) 로직 동일(diff = 주석 언어뿐).

---

## 3. ML 관련 리뷰 코멘트 — 기술적 대응 방안

우선순위·의존관계 순. (전체 코멘트 대응표는 이전 정리 참조; 여기는 ML 항목만 심화)

### R2 #6 — 클래스별 sensitivity/specificity/precision/recall/F1 [대응: 재학습 + 재계산]

**현재 상태의 기술적 문제:**
- 최종 파이프라인은 지표를 저장하지 않음 (summary.json = 카운트만).
- 현 val 세트(3그룹/171윈도우)는 **Rhonchi 0개, Non-breathing 87/171**로 퇴화 → 여기서 계산한 지표는 무의미.
- 그룹 수 자체가 24개뿐이라 `test_size=0.10`이면 val이 3그룹 — 어떤 시드든 클래스 결손 위험 높음.

**대응 설계:**
1. **분할 재설계**: 클래스-그룹 동시 고려 분할 — `StratifiedGroupKFold`(sklearn ≥1.0, 그룹=녹음 stem)로 k=5 CV. 24그룹/5클래스 규모에서는 단일 홀드아웃보다 **CV 평균±표준편차 보고**가 통계적으로 유일하게 방어 가능.
2. 지표: 클래스별 recall(=sensitivity), specificity(OVR TN 기반), precision, F1 + macro/weighted 평균 + confusion matrix. 클래스 불균형(52.5% Non-breathing) 때문에 accuracy 단독 보고 금지.
3. 신뢰구간: 그룹 단위 bootstrap 또는 fold 간 분산으로 제시하면 R2·R3 동시 방어.

### R3 Major #2 — Non-breathing 혼동, FP/FN 알림율, 클래스 정의 [대응: 재분석 + 서술]

1. **Non-breathing 정의는 코드에서 명확**: 전문의가 주석한 흡기/호기 구간의 **여집합**(호흡 사이 무음·환경음·핸들링 노이즈). 이 정의를 본문에 명시 (breath-holding, 말소리 등 별도 수집 아님도 정직하게).
2. **윈도우 지표 → 알림 지표 변환**: 리뷰어가 원하는 것은 window-level confusion이 아니라 alert-level FP/FN. R1 #4의 집계 규칙(아래)을 정의한 뒤, 30 s 녹음 단위로 "비정상 알림 발생 여부" vs ground truth를 재집계해 **alert-level FP rate / FN rate** 산출. 스크립트는 `tetho_softmax_export.py`의 배치 추론 출력을 재활용하면 됨.
3. Non-breathing recall 개선 여지: coverage 경계(호흡↔비호흡 전이에 걸친 윈도우)가 혼동의 주원인일 가능성 → coverage 임계 상향 또는 전이 윈도우 제외 재실험으로 정량화 가능.

### R1 #4 — 윈도우 단위 알림 과다 → 집계·알림 기준 [대응: 재구현]

앱은 현재 알림 트리거 자체가 미구현이므로(§4.8) 이번에 설계·구현:
- **후보 규칙**: (a) K-of-N 룰 — 최근 N=5 윈도우 중 K=3 이상 동일 비정상 클래스, (b) EMA 스무딩 p̃ₜ = αpₜ + (1−α)p̃ₜ₋₁ 후 임계 초과 지속시간 조건, (c) 호흡 주기 동기(에너지 엔벨로프 기반 사이클당 1판정).
- 히스테리시스(발동 임계 > 해제 임계)로 채터링 방지.
- 답변에는 선택 규칙 + 시뮬레이션 결과(기존 녹음 재생 시 알림 횟수 before/after)를 제시.

### R2 #8 / R3 Minor #3 — 레이턴시 [대응: 측정 + 코드 일원화]

1. 선행: 홉 일원화(논문 0.5 s vs 앱 1.0 s vs export 0.5 s → 하나로 통일, §4.5).
2. 측정 항목 분해: BLE 패킷 수신→버퍼 완성(구조상 최대 = 윈도우 2 s + 홉), 특성 추출(ms 단위), scaler+LR+MLP 추론(ms), BLE 액추에이션 명령 왕복. `time.perf_counter()` 계측 코드 삽입 → 평균/최대/분포 보고.
3. 클래스 간 지연 차이 없음(단일 경로)도 명시 가능 — R2가 물어봄.

### R2 #9 — 고정 2 s 윈도우 / 패딩 [대응: 서술 정정 + 영향 분석]

1. **서술 정정 필수**: 논문의 "padding applied to shorter segments"는 사실이 아님 — 실제는 2 s 미만 꼬리 폐기 + coverage 기반 라벨. 패딩 인공 패턴 우려는 "패딩을 쓰지 않으므로 해당 없음"으로 답하되 폐기 방식의 손실을 정량화.
2. 영향 분석: 호흡 구간의 88.8%가 2 s 미만 → 0.5 s 홉 중첩으로 짧은 이벤트가 몇 개 윈도우에 포착되는지 커버리지 통계(구간 JSON에서 직접 계산 가능) 제시.
3. Discussion의 adaptive segmentation(L487–490) 향후 방향과 연결.

### R2 #5 — 세트별 참가자·녹음 수 [대응: 재기술 (선행: §4.3 해소)]

실측: RAW 38 wav(34환자) 중 주석 보유 24녹음(21환자)만 특성화, train 21그룹(1,197윈도우)/val 3그룹(171윈도우). §3의 재학습 후 fold별 그룹·윈도우 수 표로 제시.

### R2 #3 — 라벨링 절차 [대응: 설명 + 변환 규칙 명문화]

임상 측(전문의 수, 독립성, 불일치 해소) 확인과 별개로, 우리 쪽 기여: **주석→학습 라벨 변환 규칙**을 명문화 — 녹음 단위 진단(dominant sound)이 해당 녹음의 모든 breathing 윈도우에 전파되는 구조(윈도우 단위 재청취 아님). 리뷰어의 "dominant sound인지, any sound인지, 진단인지" 질문에 대한 정확한 답은 "녹음 수준 dominant/진단 라벨의 구간 전파"임.

### R1 #1 / R2 #1 — 진단 아님 [대응: 서술]
모델 출력은 소리 패턴 분류 확률임을 명시. `argmax(p−τ)`도 "diagnosis"가 아니라 "sound-class decision"으로 재명명.

---

## 4. 논문-코드 정합성 전수 감사 결과 (실행 검증 완료)

**방법**: 코드 정독 + npz/pkl/json 로드, 분할 재현, 배포 pkl로 성능·임계값 재계산, 앱 결정 로직 시뮬레이션. 아티팩트 4종은 학습 폴더·배포 폴더·논문 사본에서 **md5 동일** → 재계산 결과는 논문 모델에 그대로 적용됨.

### 4.1 🔴 Fig 4B(≥85%)의 산출 근거가 로컬에 없음
- 최종 파이프라인은 성능 지표 미저장(스크립트가 test/지표/이미지 로직을 의도적으로 제거 — 헤더 "(변경 요약)" 참조).
- 프로젝트 전수 탐색 결과 어떤 confusion matrix도 Fig 4B와 불일치:

| 후보 | 위치 | accuracy / 대각 | Fig 4B와 |
|------|------|----------------|----------|
| Fig 4B (논문) | — | 대각 89/85/87/94/65 (H/W/C/R/NB) | 기준 |
| 이전 버전 test | `2 Model Training …/result/run_20251102_225046` | 63.2% (C 100/H 45.5/NB 56.4/R 68.8/W 81.2) | ✗ |
| Backup Plan | `Deployment/Backup Plan/result` | 87.7% | ✗ (대각 상이) |
| Group_Split | `Deployment/Group_Split/result/run_2025100x` | 79.7% ×3 | ✗ |
| 최종 run 자체 val 재계산 | 배포 pkl 직접 | 47.95% (단 퇴화 val — 아래 4.2) | ✗ |

- **액션**: 1저자에게 Fig 4B의 데이터·평가 스크립트 출처 확인. 확보 불가 시 §3의 재학습 결과로 Fig 4B 재산출이 유일한 방어선.

### 4.2 🔴 Rhonchi 임계값 = ∞ → 앱에서 Rhonchi 진단 영구 억제 (실동작 결함)
- **메커니즘 (실행 재현으로 확정)**: val 3그룹(H001, KP007, KP017_1)에 Rhonchi 표본 0 → `sklearn.roc_curve`가 양성 없는 클래스에 thresholds 배열 선두 `inf` 반환 → Youden J argmax가 inf 지점 선택 → `thresholds.json`에 `Infinity` 직렬화 → 앱 `argmax(p − τ)`에서 Rhonchi 점수 = p − ∞ = −∞ → **어떤 입력에도 Rhonchi 선택 불가** (시뮬레이션: Rhonchi 확률 0.6이어도 Crackle로 판정).
- 부수: UI가 Rhonchi를 "Bronchi"로 표기해 관찰로 발견 어려움.
- **수정**: (a) 임계값 산출 시 `np.isfinite` 가드 + 양성 결손 클래스는 τ=0.5 폴백, (b) 근본적으로는 §3 층화 재분할로 결손 자체 방지, (c) 앱 로드 시 finite 검증 추가.

### 4.3 🟠 "38명 학습" vs 실측 21환자/24녹음/1,368윈도우
- RAW 38 wav 중 14개(WEBSS 11, KP002, KP012 2세션)는 breathing 구간 주석 부재로 파이프라인에서 자동 제외.
- Table S2(38명)는 **수집** 명세이지 **학습 사용** 명세가 아님 — 본문 L357 "38 patients were included for model training"과 상충. 제외 사유(주석 미완)와 함께 재기술 필요.

### 4.4 🟠 "train/validation/test" (L352–353) vs 실제 9:1 train/val
- test 세트·지표는 이전 버전(2번 폴더, run_20251102: 72/8/20 분할)에만 존재. 최종 모델에는 test가 없으므로 논문의 3-세트 서술 유지 불가 → §3 CV 체계로 대체 서술 권장.

### 4.5 🟠 실시간 업데이트 주기: 논문 0.5 s vs 앱 `HOP_S=1.0`
- 학습 윈도우 생성 0.5 s / 배포 앱 1.0 s / export 0.5 s — 코드 간에도 상충. 추론 자체는 홉과 무관하게 유효하나(윈도우 독립 추론), 레이턴시 보고(R2#8) 전에 반드시 일원화.

### 4.6 🟠 "padding applied" vs 패딩 부재 (2 s 미만 폐기)
- np.pad/tile/fix_length 계열 전무. §3 R2#9 참조.

### 4.7 🟡 계층 분류 서술(L158–160) vs 플랫 5-클래스 구현
- 논문 내부에서도 L374–376(플랫 그룹핑)과 상충. L158–160을 플랫 서술로 통일.

### 4.8 🟡 HR/BR 기반 알림(L385–386) 미구현
- 앱에 HR/BR 산출·비교·알림 코드 없음(bpm 카드 정적 "--", alert dot `set_on` 미호출). HR/BR은 MATLAB 오프라인 분석(L583–638)에만 존재. 구현하거나 서술에서 제거.

### 4.9 🟡 ">70% 확신도"(L394) 게이트 부재
- 앱은 확률 표시만. `argmax(p−τ)` 결과는 폐기. 대표 사례 서술로 한정하거나 게이트 구현.

### 4.10 🟡 주파수 대역: fmax 7900 vs 실효 나이퀴스트 2 kHz
- §2.1 참조. 성능 중립(학습=추론 조건)이나, 논문의 150–2000 Hz / 200–1900 Hz 서술과 특성 설정의 관계를 정리해 두어야 리뷰어 후속 질문에 방어 가능. 재학습 시 fmax를 2000 이하로 낮추고 n_mels 재배치하는 것이 기술적으로 정직한 선택.

### 정합성 PASS (문제 없음 확인)
특성 파라미터 학습↔앱 전 항목 일치(SR/n_mels/fmin/fmax/win/hop/정규화) · 클래스 이름 매핑과 재정렬 성립 · 그룹 누수 0 (세션 접미어 제거 후 환자 수준도 0) · 앙상블 0.5:0.5 일관 · 아티팩트 E2E 추론 정상(확률 합 1, NaN 없음) · 앱 3버전 로직 동일.

---

## 5. 권장 액션 플랜 (의존관계 순)

| 순서 | 작업 | 유형 | 해소되는 항목 |
|------|------|------|--------------|
| 1 | 1저자에게 Fig 4B 산출 데이터·스크립트 확인 | 확인 | §4.1 |
| 2 | StratifiedGroupKFold 재학습 + 클래스별 지표·CI 산출 + finite 임계값 가드 | **재학습/재구현** | §4.1(대안), §4.2, R2#5, R2#6 |
| 3 | 알림 집계 규칙 구현(K-of-N/EMA) + alert-level FP/FN 재계산 | **재구현/재분석** | R1#4, R3 Major#2 |
| 4 | 앱 정비: 홉 일원화, Rhonchi 억제 수정, (필요 시) HR/BR·확신도 게이트 구현 여부 결정 | **재구현** | §4.2, §4.5, §4.8, §4.9 |
| 5 | 레이턴시 계측 코드 삽입 → 평균/최대 보고 | 측정 | R2#8, R3 Minor#3 |
| 6 | 원고 서술 정정: 패딩→폐기, 3세트→CV, 38명→실사용, 계층→플랫, 주파수 대역, 진단→분류 용어 | 서술 | §4.3–4.7, §4.10, R1#1, R2#1, R2#9 |

> 2–4번은 하네스(`stetho-orchestrator`)로 실행 가능: ml-trainer(재학습·지표) → qa-evaluator(검증) → deployment-engineer(앱 수정) 순.
