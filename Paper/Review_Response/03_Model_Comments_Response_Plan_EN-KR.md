# Response Plan for Model-Related Reviewer Comments (영문 + 한글 해석)

**Manuscript:** "A Soft Wearable System for Real-time Respiratory Disease Diagnosis and Vibrotactile Intervention" (Science Advances)
**Prepared by:** Yun Tae Hwang — 2026-08-14 (rev. 3; scope narrowed to genuinely model-technical items, verified against the final code over two internal audit rounds)

> **[한글 해석]** 모델 관련 리뷰어 코멘트 대응 계획. rev.3은 "진짜 모델 기술 항목"으로만 범위를 좁힌 버전이고, 모든 내용은 최종 코드에 대해 내부 감사를 2회 돌려 검증한 사실에 기반한다.

## Executive Summary

No new data collection and no new clinical experiments are required to address the model-related comments. Six comments fall within the genuinely model-technical scope — the questions that can only be answered from the training pipeline and the classifier itself — and each of them can be resolved either by literature-backed justification, by re-analysis of data we already have (computation only), or by clarifying the Methods text. The remaining comments that mention the model only in passing (sound-versus-diagnosis framing, prevention claims, labeling procedure) can be answered generally without model work; for those I provide one-line technical inputs where useful, and they are otherwise outside this plan.

The proposed timeline is **two weeks** from confirmation of the finalized code set. Two prerequisites — one external, one internal — should be settled early in Week 1: the clinical team's one-line confirmation of the annotation procedure (needed for the labeling answer, whoever writes it), and an internal decision to unify the real-time update interval between the manuscript and the app (needed before the latency answer is written; details in the internal memo). Week 1 covers the code-set verification, the per-class re-evaluation, the alert-level analysis, and the latency bench measurement; Week 2 covers the point-by-point responses and revised manuscript text, with internal review by Dr. Oh before delivery to Prof. Yeo.

> **[한글 해석]** 핵심 요약 — 교수님의 두 관심사에 대한 답:
> 1. **추가 실험 필요 없음.** 새 데이터 수집이나 임상 실험은 불필요하다. 모델 기술 항목은 딱 6개이고(학습 파이프라인과 분류기에서만 답이 나오는 질문들), 각각 ① 문헌 근거 서술, ② 이미 있는 데이터의 재분석(계산만), ③ Methods 문구 수정 중 하나로 해결된다. 모델을 스치듯 언급하는 나머지 코멘트(소리 vs 진단, 예방 주장, 라벨링 절차)는 모델 작업 없이 일반적 답변으로 처리 가능 — 필요한 곳에 한 줄짜리 기술 입력만 제공한다.
> 2. **타임라인 2주** (최종 코드 세트 확정 시점부터). 단, 1주차 초반에 해결해야 할 선행조건이 둘 있다: (외부) 임상팀의 라벨링 절차 확인 한 줄, (내부) 본문과 앱 사이에 어긋나 있는 실시간 업데이트 주기 통일 결정. 1주차에 코드 검증·클래스별 재평가·알림 수준 분석·레이턴시 측정, 2주차에 답변서와 수정 원고 작성 후 박사님 내부 리뷰를 거쳐 교수님께 전달.

---

## Core Model-Technical Items (our scope)

> **[한글 해석]** 아래 6개가 "진짜 모델 기술" 항목 — 우리(모델 담당)가 직접 처리해야 하는 범위다.

### R2-9 — Why 2-s windows and a 0.5-s hop; what about split events and padding

The reviewer worries that fixed windows may divide a respiratory event between segments or miss short events, and that padding shorter segments may introduce artificial patterns. This is precisely the "why did you set it up this way" class of question, and the answer is a single grounded argument rather than a new experiment.

The window length is defensible from physiology and from our own annotations. An adult respiratory rate of 12–20 breaths per minute corresponds to a 3–5-s cycle, which at a typical 1:2 inspiratory-to-expiratory ratio gives roughly 1–2 s per phase, so a 2-s window is long enough to encompass a complete inspiratory or expiratory phase together with the transient adventitious events inside it. Our annotated data agree: 88.2% of breath phases are shorter than 2 s with a median around 1.2 s — shorter than the adult theoretical bound, consistent with the faster respiratory rates of the pediatric asthma participants. The choice is also in line with the 1–3-s analysis windows commonly used in the lung-sound classification literature, and we will attach references to that effect. For faster breathing, where a window may span more than one phase, the 0.5-s hop provides 4× overlap in the analysis pipeline, so each event is represented across several overlapping windows rather than at a single boundary.

On padding, the Methods wording needs a clarification: upon further investigation of the final implementation, residual sub-2-s tails at segment boundaries are discarded rather than zero-padded, and the short events they might contain are still captured by the neighboring overlapping windows. We will revise the Methods sentence accordingly and note that no artificial zero-padded patterns enter the model input. The response can close by pointing to the Discussion's existing statement that adaptive segmentation is a direction for future work.

> **[한글 해석]** R2-9 — 왜 2초 윈도우/0.5초 홉인가, 이벤트 분할과 패딩 문제.
> 리뷰어의 우려는 고정 윈도우가 호흡 이벤트를 자르거나 짧은 이벤트를 놓치고, 패딩이 인공 패턴을 만들 수 있다는 것. 이건 박사님이 말씀하신 "왜 이렇게 설정했냐" 류의 전형이라, 새 실험이 아니라 근거 있는 논증 하나로 답한다.
> **윈도우 길이 근거**: 성인 호흡수 12–20회/분 → 주기 3–5초 → 흡기:호기 1:2 비율에서 phase당 약 1–2초. 즉 2초 윈도우면 흡기 또는 호기 한 phase와 그 안의 이상음 이벤트를 통째로 담는다. 실측도 일치: 우리 주석 데이터에서 phase의 88.2%가 2초 미만(중앙값 약 1.2초)이고, 이론값보다 짧은 건 소아 천식 참가자의 빠른 호흡 때문 — 소아 코호트를 리뷰어보다 먼저 언급해 선제 방어한다. 폐음 분류 문헌의 통상 윈도우(1–3초)와도 부합하며 레퍼런스를 붙일 예정. 호흡이 빨라 윈도우 하나에 여러 phase가 걸리는 경우는 0.5초 홉의 4× 오버랩이 각 이벤트를 여러 윈도우에 걸쳐 잡아준다.
> **패딩**: "추가 검토 결과(upon further investigation)" 프레임으로 — 실제 구현은 2초 미만 꼬리를 제로패딩이 아니라 **폐기**하고, 짧은 이벤트는 이웃한 중첩 윈도우가 잡는다. Methods 문장을 이렇게 정정하고, 따라서 인공 패딩 패턴은 모델 입력에 들어가지 않는다고 답한다. 마무리는 Discussion에 이미 있는 adaptive segmentation 향후 과제로 연결.

### R2-6 — Per-class sensitivity, specificity, precision, recall, F1

The reviewer is right that a single accuracy figure hides per-class behavior. We will re-evaluate the already-extracted features with patient-grouped repeated evaluation — recordings from the same participant never split across training and evaluation, repeated so that variability can be reported — and provide per-class sensitivity, specificity, precision, F1, and the full confusion matrix as mean ± SD. The class-decision thresholds will be re-derived within the same run with an explicit finite-value guard for classes that are sparse in an evaluation split. This is computation on existing features; no new recordings are involved. The specific confusion figures that Reviewer 3 quotes (non-breathing misread as crackle, wheezing, or healthy) will be re-derived in the same run so that every number in the response is mutually consistent.

> **[한글 해석]** R2-6 — 클래스별 지표(민감도/특이도/정밀도/재현율/F1).
> accuracy 하나로는 클래스별 성능이 안 보인다는 지적은 맞다. 이미 추출된 특성으로 **환자 그룹 단위 반복 평가**(같은 참가자의 녹음이 학습/평가에 나뉘지 않게, 여러 번 반복해 변동성까지 보고)를 돌려 클래스별 지표와 confusion matrix를 평균±표준편차로 제시한다. 임계값도 같은 실행에서 재산출하되, 평가 분할에 샘플이 희소한 클래스에 대해 유한성 가드를 명시적으로 넣는다(현재 Rhonchi 임계값 무한대 문제의 재발 방지 — 내부 메모 참조). 새 녹음 없이 기존 특성 계산만이다. R3가 인용한 구체 혼동 수치(non-breathing→crackle 등)도 같은 실행에서 재산출해 답변서의 모든 숫자를 일관되게 만든다.

### R2-5 — Exact participants and recordings per data split

This looks like a writing item but the numbers must come from the pipeline, so it rides on the R2-6 run. We will produce an explicit table — participants, recordings, and analysis windows per split and per class — directly from the re-evaluation output, and align the Methods text with it, including the distinction between enrolled participants, participants whose recordings entered model training, and the per-split breakdown. The dataset-description sentences will be updated to match the table exactly.

> **[한글 해석]** R2-5 — 분할별 정확한 참가자·녹음 수.
> 겉보기엔 서술 항목이지만 숫자가 파이프라인에서 나와야 하므로 R2-6 실행에 얹어서 처리한다. 분할별·클래스별 참가자/녹음/윈도우 수 표를 재평가 산출물에서 직접 생성하고, Methods 본문을 이 표와 정확히 일치시킨다. 이때 "등록 참가자 수"와 "실제 학습에 들어간 참가자 수"의 층위를 구분해서 서술한다 (내부적으로는 43명 등록 / 본문 38명 / 실투입 21명·24녹음의 3층 구조 — 상세는 내부 메모).

### R3-Major2 — What the non-breathing class is, and what its confusion means for alerts

Two things are being asked: a definition and a consequence. The definition is precise in the pipeline: non-breathing is the complement of the physician-annotated breath phases within each recording — inter-breath silence and ambient background. Speech, motion artifact, and breath-holding were not separately curated, and we will say so explicitly as a limitation; a class that is heterogeneous by construction is expected to show lower single-window separability.

The consequence is about alerts, not windows, and that is where the answer gains ground: actuation decisions are not made per window. With the aggregation rule described under R1-4, transient window-level confusions are suppressed unless they persist across consecutive windows. We will quantify this by re-running the existing recordings offline and reporting alert-level false-positive and false-negative rates at the recording level — the clinically relevant quantity for actuation decisions — alongside the window-level confusion matrix from R2-6.

> **[한글 해석]** R3-Major2 — non-breathing 클래스의 정의와, 그 혼동이 알림에 갖는 의미.
> 질문은 둘: 정의와 결과. **정의**는 코드상 명확하다 — 전문의가 주석한 호흡 phase의 여집합(호흡 사이 무음 + 배경음). 말소리·동작 잡음·숨참기를 따로 수집하진 않았다고 한계로 솔직히 명시한다. 구성상 이질적인 클래스라 단일 윈도우 분리도가 낮은 건 예상 가능한 일이라는 논리로 65% 수치를 방어한다.
> **결과**는 윈도우가 아니라 알림 차원의 문제인데, 여기서 반격의 여지가 생긴다 — 액추에이션 결정은 윈도우 단위가 아니다. R1-4의 집계 규칙 하에서는 일시적 윈도우 혼동이 연속으로 지속되지 않는 한 억제된다. 기존 녹음을 오프라인 재실행해서 **녹음 수준의 알림 오경보/미탐지율**(액추에이션 관점에서 임상적으로 유의미한 수치)을 R2-6의 윈도우 confusion matrix와 나란히 제시한다.

### R1-4 — Aggregating per-window predictions into clinically meaningful alerts

The reviewer's concern is over-alerting and habituation if every abnormal 2-s window triggers feedback. We will define a simple aggregation rule — an alert fires only when a majority of recent windows (for example K of the last N) agree on the same abnormal class — and demonstrate its effect by re-running existing recordings offline and reporting alert counts before and after aggregation. This is a simulation on data we already have. The clinical meaning of each tactile cue is the actuation team's part of this comment and is handled with R3-Major3.

> **[한글 해석]** R1-4 — 윈도우 예측을 임상적으로 의미 있는 알림으로 집계.
> 비정상 윈도우마다 피드백이 울리면 과잉 경고·습관화가 생긴다는 우려. 간단한 집계 규칙(최근 N개 윈도우 중 K개 이상이 같은 비정상 클래스일 때만 알림 발동)을 정의하고, 기존 녹음을 오프라인 재실행해 집계 전/후 알림 횟수를 비교해 보여준다. 이미 있는 데이터로 하는 시뮬레이션이지 새 실험이 아니다. 각 촉각 큐의 임상적 의미 부분은 액추에이터 팀 몫(R3-Major3과 함께 처리).

### R2-8 & R3-Minor3 — Processing latency

Once the update interval is unified between the manuscript and the app (the internal prerequisite above), we will instrument the existing real-time app with timing probes at each stage and report mean and maximum latency from bench playback. Structurally, end-to-end latency is dominated by the 2-s analysis window plus the update interval; the feature-extraction and inference path is a single 64-dimensional vector through the same scaler and ensemble for every class, so there is no class-dependent delay, and we will state that explicitly since the reviewer asked. This is a bench measurement, not a clinical experiment.

> **[한글 해석]** R2-8 & R3-Minor3 — 처리 레이턴시.
> 선행조건인 업데이트 주기 통일(본문 0.5초 vs 앱 1초 — 내부 메모 §3)이 끝나면, 기존 앱에 구간별 타이밍 계측을 넣어 벤치 재생으로 평균/최대 레이턴시를 보고한다. 구조적으로 레이턴시는 2초 분석 윈도우 + 업데이트 주기가 지배하고, 특성 추출·추론 경로는 모든 클래스가 동일한 64차원 벡터 → 동일한 scaler/앙상블을 타므로 **클래스 간 지연 차이는 없다** — 리뷰어가 물었으니 명시적으로 답한다. 벤치 측정이지 임상 실험이 아니다.

---

## Items We Only Support (general answers, no model work)

Three comments touch the model but do not require model work, and per our discussion they can be answered generally. For the sound-versus-diagnosis comments (R1-1 first part, R2-1), the technical input from my side is one sentence: the classifier outputs sound-class probabilities, not disease labels, so the terminology revision ("diagnosis" → "respiratory sound classification") is accurate to the implementation. The same claim-softening pass covers the prevention-claims comment (R2-2). For the labeling-procedure comment (R2-3), the clinical team must confirm the number of annotating physicians and the adjudication procedure; the one model-side fact worth including in that answer is that labels were assigned per recording (dominant adventitious sound with the clinical diagnosis) and propagated to all breathing windows of that recording during segmentation — individual windows were not re-auscultated, which directly answers the reviewer's dominant-sound/any-sound/diagnosis question.

> **[한글 해석]** 우리가 "지원만" 하는 항목 — 모델 작업 없이 일반 답변으로 처리 가능한 3건 (통화에서 말씀하신 "바이패스" 영역).
> - **R1-1 전반부·R2-1 (소리 vs 진단)**: 우리 쪽 기술 입력은 한 문장 — "분류기 출력은 질병 라벨이 아니라 소리 클래스 확률이므로, diagnosis → respiratory sound classification 용어 수정이 구현과 정확히 일치한다."
> - **R2-2 (예방 주장)**: 같은 주장 완화 작업으로 커버됨.
> - **R2-3 (라벨링 절차)**: 전문의 수·판정 절차는 임상팀 확인 사항. 우리 쪽에서 답변에 넣을 사실 하나는 "라벨은 녹음 단위로(dominant 이상음 + 임상 진단) 부여되어 세그먼트 단계에서 그 녹음의 모든 breathing 윈도우에 전파됐고, 윈도우 개별 재청취는 없었다" — 리뷰어의 dominant/any/진단 3택 질문에 대한 직답이다.

## Out of Scope (other co-authors)

R1-1 second part (behavioral/clinical benefit of feedback), R1-2 (HR/BR gold-standard validation), R1-3 (power/battery), R1-5–7 and R3-Major3/4 (tactile-cue rationale, human-subject validation), R2-4 (demographics), R2-7 (introduction references), R2-10 (patient response to alerts), R3-Minor1 (MEMS spec), R3-Minor2 (cavity simulation), R3-Minor4 (long-term home use). R3-Major1 (sensor placement) is a clinical-protocol item; its generalizability implication connects to the existing Discussion limitation on user-specific variability.

> **[한글 해석]** 타 공저자 담당 (우리 범위 밖): R1-1 후반부(피드백의 행동적/임상적 이득), R1-2(HR/BR 기준기기 검증), R1-3(전력/배터리), R1-5~7·R3-Major3/4(촉각 큐 근거·인체 검증), R2-4(인구통계), R2-7(서론 문헌), R2-10(알림에 대한 환자 반응), R3-Minor1(MEMS 사양), R3-Minor2(음향공동 시뮬), R3-Minor4(장기/가정 사용). R3-Major1(센서 위치)은 임상 프로토콜 항목이되, 그 일반화 함의는 Discussion의 기존 한계 서술(사용자별 변동성)에 연결해 처리.

## Summary for the Meeting

| Question | Answer |
|---|---|
| New experiments needed? | **No.** Two computational re-analyses on existing data (R2-6 metrics, R1-4/R3-Major2 alert-level simulation) plus one bench measurement (latency); everything else is literature-backed writing and Methods clarification. |
| Timeline | **2 weeks** from code-set confirmation, with two Week-1 prerequisites: CNUH annotation-procedure confirmation (external) and update-interval unification (internal). |

> **[한글 해석]** 미팅용 요약 —
> **추가 실험?** 없음. 기존 데이터 재분석 2건(R2-6 지표, R1-4/R3-Major2 알림 시뮬레이션) + 벤치 측정 1건(레이턴시)뿐이고 나머지는 문헌 근거 서술과 Methods 정정.
> **타임라인?** 코드 세트 확정 후 2주. 1주차 선행조건 2개: CNUH 라벨링 절차 확인(외부), 업데이트 주기 통일(내부).
