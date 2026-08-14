# Round 2 — 적대적 재검증 감사 (Adversarial Re-verification)

- **목적**: 이전 감사(`04_qa_paper_code_audit.md`)의 모든 사실 주장을 "틀렸을 가능성"을 전제로 독립 재실행하여 반박(refute)을 시도.
- **방법**: 코드 재정독 + 실제 파이썬 실행(npz/pkl 로드, 분할 재현, val 성능·임계값 재계산, 앱 결정 로직 시뮬레이션, 구간 분포 계산, 전체 프로젝트 confusion matrix 스캔). 파일 미수정.
- **검증 스크립트** (재실행 가능):
  - `_workspace/qa_scripts/round2_01_npz_split_perf.py` — npz 검사, 환자/녹음/윈도우 카운트, 분할 재현, 누수, 임계값·val 성능 재계산, argmax(p−τ) 시뮬레이션
  - `_workspace/qa_scripts/round2_02_intervals_overlap_sr.py` — 호흡 phase 길이 분포, 오버랩 배수, RAW SR, 제외 14파일
  - `_workspace/qa_scripts/round2_03_artifact_e2e.py` — 아티팩트 4종 end-to-end 더미 추론
  - `_workspace/qa_scripts/round2_04_hunt_fig4b.py` — 프로젝트 전체 57개 CM + 분류 리포트에서 Fig 4B(89/85/87/94/65) 탐색
- **무결성 재확인 (md5)**: scaler/lr/mlp/thresholds 4종 + features_64mel.npz + script5가 `5 Realtime Pipeline_Final`, `6 Final Deployment to Dr.Oh`, `Paper/Final_Code_Manuscript` 3곳에서 **완전 동일**. 앱은 `6 Deploy` == `Paper 사본` 동일, `4 New`는 주석만 상이(로직 동일). → 재계산 결과는 배포·논문 모델에 직접 적용됨.

**총평: 이전 감사의 10개 핵심 주장 중 반박에 성공한 것은 0건.** 9건은 실행으로 **확정(재현 성공)**, 1건(#2)은 골자는 확정이나 "제외 사유" 표현에 **미세 수정 필요**. 대응 문서(응답 플랜)의 수치 2건(오버랩 배수 적용 대상, 88.8%)에 정밀화가 필요.

---

## 주장별 판정표

| # | 주장 요지 | 판정 | 실행 근거 (수치) |
|---|-----------|------|------------------|
| 1 | Rhonchi τ=Infinity; 원인=val Rhonchi 0개; 앱 argmax(p−τ)에서 Rhonchi 절대 선택 불가 | **확정** | 배포 thresholds.json Rhonchi=inf; val 분할 재현 시 val 3그룹(H001·KP007_WWS·KP017_WWS_1) Rhonchi **0개**; roc_curve 양성 0 → thr=inf 재현; argmax(p−τ) 171 val 윈도우 전체에서 Rhonchi 선택 횟수 **0 (YES/NO=NO)** |
| 2 | 학습 실사용 21환자/24녹음/1,368윈도우; RAW 38 중 14 제외(주석 부재) | **확정(수치)** / **미세 수정(제외 사유)** | npz N=**1368**, 그룹=**24**, 환자=**21**(세션접미어 _1/_2 제거해도 21, 첫 토큰 기준도 21 — 일치); RAW=**38**(전부 4000Hz); 제외 정확히 **14개**={KP002_WWS, KP012_WWS_1, KP012_WWS_2, WEBSS×11}. **단**: KP002/KP006/KP009/KP017_2는 intervals JSON에 **키는 존재**하나 `Segments_from_JSON`에 세그먼트가 없어 features에서 빠짐 → "주석 부재"보다 "세그먼트 추출 단계(스크립트 2) 산출물 부재/파일명 세션접미어 불일치"가 정확 |
| 3 | 짧은 세그먼트 패딩이 코드 어디에도 없음 | **확정** | 전 스크립트 grep: `np.pad/tile/fix_length/zero-pad/constant_values` **전무**. 유일한 "pad" 토큰은 앱 CSS `padding`(스타일). 세그먼트 생성(`3:104,196`)·export(`tetho:131`) 모두 `if i1>n: break`로 2초 미만 꼬리 폐기. 스크립트 헤더 "패딩 제거" 명시 |
| 4 | 배포 앱 HOP_S=1.0, export=0.5 | **확정(+추가 모순)** | 앱 `HOP_S=1.0`(L40); export `HOP_S=0.5`(L28)이나 **모든 주석이 "1s hop"**(L28·117·154·226 제목). 게다가 L120 주석 "16000 samples"는 1s용, 실제 `int(0.5*16000)=8000`(0.5s) → export 내부 코드·주석 다중 모순 |
| 5 | 모델은 플랫 5클래스(계층 아님) | **확정** | LR·MLP 각각 `classes_=[0,1,2,3,4]`, `n_features_in_=64`, `P=0.5·LR+0.5·MLP` 단일 5-way. 계층 분기 코드 없음. 앱 `_infer_one`·export 모두 단일 벡터 |
| 6 | 앱에 HR/BR 계산·알림 트리거·70% 게이트 없음 | **확정** | `card_hr/card_br`="-- bpm" 정적 라벨(L347-348), bpm/peak/rate 산출 코드 없음; `set_on()` 호출 0회(정의만); `0.7/70%` 게이트 grep 0건; `get_latest_label_ui()` 반환값 `_ =`로 폐기(L401) |
| 7 | 최종 산출물에 성능지표 없음 + 프로젝트 어떤 CM도 Fig 4B(89/85/87/94/65)와 불일치 | **확정** | summary.json=카운트만(성능 무). **프로젝트 전체 CM 57개 CSV + 모든 classification_report 스캔** → Fig 4B 대각(정렬 65/85/87/89/94)과 ±3pp 이내로 일치하는 결과 **0건**. Non-breathing 포함 5클래스 후보 다수(아래 §목록) 나열했으나 전부 불일치 |
| 8 | val 재계산 정확도 47.95% (배포 pkl 직접) | **확정** | 배포 pkl 직접 로드 → val argmax **47.95%**; recall: NB 90.8%, Wheezing 6.7%, Crackle 0%, Healthy 0%, Rhonchi N/A(0표본). 재현 임계값이 배포 thresholds와 5/5 완전 일치 → 동일 run 확정 |
| 9a | "0.5s hop → 4× overlap" | **확정(단, 적용 대상 주의)** | 2.0/0.5 = **4.0×** 맞음. **그러나 이는 학습·export(0.5s hop)에만 해당**. 배포 앱은 1.0s hop → **2.0×**. 대응 문서가 "우리 시스템"의 오버랩으로 4×를 제시하면 배포 앱과 불일치 |
| 9b | "대다수 호흡 phase가 2초 미만" (기존 88.8%) | **확정(수치 정밀화)** | intervals JSON 29녹음/330 phase: **<2s = 88.18%** (291/330). <1.3s=58.8%, <1.0s=37.3%, median 1.155s. → "88.8%"는 근사치이며 정확값은 **88.2%** |
| 9c | 추론 경로가 클래스와 무관하게 단일 | **확정** | 단일 5-way 앙상블, 클래스별 분기 없음(#5와 동일 근거). 레이턴시 클래스 독립 서술은 코드와 일치 |
| 9d | 레이턴시는 2s 윈도우+홉이 지배 | **확정(구조적)** | 추론당 특성추출은 (64,) 벡터 1회 + LR/MLP 예측(경량). 결과 생성 주기는 앱 hop=1.0s(윈도우=2.0s)에 의해 결정 → 구조상 윈도우+홉 지배 서술 타당. (실측 벤치는 별도 필요) |
| 10 | 그룹키=시간범위 제거 stem; train/val 그룹 겹침 0 | **확정** | GroupShuffleSplit(test_size=0.10, rs=42) 재현: train 21그룹/1197윈도우, val 3그룹/171윈도우; **그룹 겹침 0, 환자 겹침 0**. 그룹키 정의 = `_(t0)-(t1)$` 제거 stem 확인 |

---

## 세부 실행 결과

### #1 · #8 · #10 — 분할·임계값·성능 (round2_01)
```
X shape (N x dim): (1368, 64)
class dist: Crackle 149, Healthy 91, Non-breathing 718, Rhonchi 121, Wheezing 289
unique groups: 24 ; unique patients(첫토큰): 21 ; 세션접미어 제거 후 녹음: 21
GroupShuffleSplit(test_size=0.10, rs=42):
  train 1197 win / 21 groups ; val 171 win / 3 groups
  val groups = [H001, KP007_WWS, KP017_WWS_1]
  GROUP overlap: NONE ; PATIENT overlap: NONE
  val class dist: Crackle 21, Healthy 18, Non-breathing 87, Wheezing 45 ; Rhonchi 0
deployed thresholds = [3.38e-4, 1.94e-3, 0.5088, inf, 6.82e-4]
reproduced         = [3.38e-4, 1.94e-3, 0.5088, inf, 6.82e-4]  (5/5 MATCH)
VAL argmax accuracy = 47.95%
per-class recall: Crackle 0.0 / Healthy 0.0 / Non-breathing 0.908 / Rhonchi N/A(0) / Wheezing 0.067
argmax(p−τ) 선택 분포: Non-breathing 126, Wheezing 33, Crackle 7, Healthy 5, Rhonchi 0
Rhonchi ever selected by argmax(p−τ)?  NO
```

### #2 · #9a · #9b — RAW/제외/오버랩/구간 (round2_02)
```
overlap factor WIN/HOP = 2.0/0.5 = 4.0×  (학습/export)
                        = 2.0/1.0 = 2.0×  (배포 앱)
breath phases: 29녹음 / 330 phase
  <1.0s = 37.27% ; <1.3s = 58.79% ; <2.0s = 88.18% ; >=2.0s = 11.82%
RAW wav = 38 (전부 SR=4000 Hz)
used groups = 24 ; 제외 = 14 = {KP002_WWS, KP012_WWS_1, KP012_WWS_2, WEBSS×11}
```
**#2 미세 수정 근거**: intervals JSON 키는 29개(H001~KP021)이며 KP002_WWS_1/_2·KP006_WWS·KP009_WWS·KP017_WWS_2가 features에서 빠짐. 그러나 `Segments_from_JSON`에 KP002/KP006/KP009/KP012/WEBSS 세그먼트가 **하나도 없음** → 이들은 상류 스크립트 2(Excel→세그먼트) 산출물이 없어 제외됨. 특히 KP002는 JSON엔 세션접미어 키(_1/_2)가 있으나 RAW엔 `KP002_WWS.wav`(무접미어)라 커버리지 매칭 실패 소지. 결론적으로 "주석 부재"라는 단일 사유보다 "**세그먼트 추출 단계 산출물 부재(+일부 파일명/세션접미어 불일치)**"가 정확한 표현.

### #7 — Fig 4B 탐색 (round2_04)
- 스캔 대상: confusion_matrix*.csv **57개** + classification_report*.txt/csv 전수.
- Fig 4B 목표 대각(H89/W85/C87/R94/NB65, 정렬 65/85/87/89/94)과 ±3pp 일치: **0건**.
- Non-breathing 포함 5클래스 결과물 주요 후보(전부 Fig 4B 불일치):
  | 위치 | accuracy | per-class recall(%) | Fig4B 여부 |
  |------|----------|---------------------|-----------|
  | `Deployment/Backup Plan/result/` | 87.7% | C79 H85 NB92 R83 W76 | 불일치 |
  | `Deployment/Group_Split/result/run_2025100*` (3런 동일) | — | C63 H85 NB78 R78 W90 | 불일치 |
  | `Deployment/1 Final Pipeline/2 Model Training with Replayed Sound/…run_20251102_225046` | 63.2% | C100 H45 NB56 R69 W81 | 불일치 |
  | `배포 pkl val 재계산(본 감사)` | 47.95% | C0 H0 NB91 R N/A W7 | 불일치 |
  | `Development/OPERA_.../results_step16B/…` (수십 fold) | 다양 | 모두 불일치 | 불일치 |
  | `Development/step17A_competition_augmentation/…` (fold별) | 다양(R 다수 N/A) | 불일치 | 불일치 |
- → Fig 4B와 일치하는 산출물이 프로젝트 어디에도 없다는 이전 결론 **재확정**. (Non-breathing 65%가 가장 낮은 클래스라는 패턴을 만족하는 5클래스 결과도 부재)

### #3~#6 · #9c · #9d — 코드 정합성
- 패딩 grep: 세그먼트/특성/앱/export 전부 무패딩. 유일 "pad"=CSS.
- 앱: `HOP_S=1.0`, `set_on()` 미호출, HR/BR 정적, 70% 게이트 없음, label 반환 폐기.
- export: `HOP_S=0.5` 실행값 vs 전체 주석 "1s hop" 모순(+L120 "16000 samples" 산술 오류, 실제 8000).
- E2E(round2_03): logmel 64 == scaler 64, 확률 len=5, sum=1.0, NaN 없음, 3트라이얼 정상.

---

## 대응 문서(`01_Model_Comments_Response_Plan.md`)에서 고쳐야 할 문장

> 근거: 위 재검증 결과. 문장 단위로 수정 제안.

1. **R2-9, L62** — "The 0.5-s hop yields **4× overlap** … in our data, the large majority of annotated breath phases are shorter than 2 s".
   - **수정①(오버랩 적용 대상)**: 4× overlap은 **학습/오프라인(0.5s hop)** 기준이다. **배포 실시간 앱은 HOP_S=1.0(2× overlap)**이므로 "our system yields 4× overlap"으로 서술하면 배포 코드와 불일치한다. → "The training/offline pipeline uses a 0.5-s hop (4× overlap); the deployed real-time app uses a 1.0-s hop (2× overlap)"로 분리 서술하거나, 배포 앱 hop을 0.5s로 통일 후 서술.
   - **수정②(수치)**: "large majority … shorter than 2 s"의 정확 수치는 **88.2%** (330 phase 중 291개, <2.0s). 기존 구두 수치 88.8%는 근사. 논문/응답에 수치를 명기한다면 **88.2%**로 기재.

2. **R2-9, L63 (padding)** — "sub-2-s residual segments are handled by the overlapping-window scheme rather than by zero-padding".
   - **판정: 코드와 일치(패딩 없음 확정)**. 단 "overlapping-window scheme이 sub-2s 잔여를 처리한다"는 표현은 오해 소지. 실제 구현은 **2초 미만 잔여를 폐기(break)**하며 별도 처리는 없다. → "sub-2-s residual segments at window boundaries are **discarded** (no zero-padding); overlapping windows ensure short events are still captured by neighboring 2-s windows"로 정정 권장.

3. **R2-5, L42-43 / Summary table** — "participants / recordings / windows per split".
   - 정확 카운트를 **21 참가자 / 24 녹음(세션) / 1,368 윈도우**, split=**train 21그룹·1197 / val 3그룹·171 / test 0**으로 명기. "38 participants" 류 서술이 본문에 있으면 정정. 제외 14개는 "**세그먼트 추출 단계 산출물 부재(주석/세션접미어 불일치 포함)**"로 사유 기술(단순 "주석 부재"는 부정확).

4. **R2-6 / R3-Major2, L49·L72 (per-class metrics, 재분석 계획)** — "Re-evaluate … report per-class …".
   - 현재 **배포 모델의 val 성능은 47.95%**이며 breathing 3클래스(Crackle/Healthy) recall=0, Wheezing 6.7%, **Rhonchi는 val 양성 0으로 평가 불가**. 응답에서 제시할 지표는 **재학습/재분할(그룹 인지 + 클래스 층화 교차검증) 이후 산출값**이어야 하며, "existing features로 minutes of compute"만으로 Fig 4B(≥85%)가 복원된다는 낙관은 근거가 없다. 임계값 산출 시 **inf 클램핑** 필수(현재 Rhonchi=inf가 배포됨).

5. **R2-8 / R3-Minor3, L55 (latency)** — "latency is dominated by the 2-s analysis window plus the update interval; the inference path is identical for all classes".
   - **판정: 코드와 일치(단일 경로, 클래스 독립 확정)**. update interval은 **배포 앱 1.0s**임을 명시(0.5s로 쓰면 코드와 불일치). 실측 벤치는 별도 수행 필요(구조적 서술은 유지 가능).

6. **(문서에 없으나 추가 권고)** — export 유틸(`tetho_softmax_export.py`)의 `HOP_S=0.5` 실행값과 주석("1s hop", "16000 samples")이 모순. Fig 4C(실시간 트레이스)를 이 스크립트로 생성했다면 **트레이스는 0.5s hop**로 산출된 것이며, 배포 앱(1.0s)과 다른 시간 해상도임을 인지·정정.

---

## 이전 감사 대비 회귀/변경 요약
- **반박 성공: 0/10.** 모든 핵심 주장 재현.
- **강화된 발견**: (a) export의 HOP 주석 모순에 더해 "16000 samples" 산술 오류까지 추가 확인. (b) Fig 4B 탐색 범위를 프로젝트 전체 57개 CM으로 확대했으나 여전히 0건 일치 → 결론 견고.
- **정밀화가 필요한 지점**: #2의 제외 사유 표현("주석 부재"→"세그먼트 추출 산출물 부재"), 오버랩 배수의 적용 대상(학습 4× / 앱 2×), <2s 비율 88.2%.

## 담당 배정 (수정 책임 분리)
- **논문 1저자/저자**: Fig 4B 산출 데이터·스크립트 출처 확보(#7), 참가자/녹음/split 카운트 재기술(#2), 패딩 서술 정정(#3), 오버랩·88.2% 수치 정정(#9a/#9b).
- **학습 담당**: 임계값 inf 클램핑 + 그룹인지·층화 재분할 후 per-class 지표 재산출(#1/#8, R2-6).
- **앱 담당**: HOP_S 문서·코드 일원화(앱 1.0s vs export 0.5s), export 주석 정정(#4/#6); HR/BR·알림·70% 게이트 미구현 서술 정정 또는 구현(#6).
