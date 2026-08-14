# 논문 서술 vs 실제 코드 전수 대조 감사 (QA Paper-Code Audit)

- 대상: `Paper/Final_Code_Manuscript/` (학습 스크립트 1~5 + features_64mel.npz + summary.json, 실시간 앱 + tetho_softmax_export.py, 아티팩트 4종)
- 검증 방식: 코드 정독 + 실제 파이썬 실행(npz/csv/pkl 로드, 분할 재현, val 성능·임계값 재계산, 앱 결정 로직 시뮬레이션). 코드 미수정.
- 아티팩트 무결성: scaler/lr/mlp/thresholds 4종이 `5 Realtime Pipeline_Final`, `6 Final Deployment to Dr.Oh`, 논문 사본에서 **md5 완전 동일** → 이 run이 곧 배포·논문 모델. 재계산 결과는 논문 모델에 직접 적용됨.
- 검증 스크립트: `_workspace/qa_scripts/01_inspect_npz.py`, `02_split_leakage_and_perf.py`, `03_artifact_e2e.py`

## 항목별 대조표

| # | 논문 주장 | 코드 실제 | 판정 | 근거 (파일:라인) |
|---|-----------|-----------|------|------------------|
| 1 | 30초 녹음의 흡기/호기 주석 파싱 → breathing/non-breathing 변환 | Excel의 흡기/exhale 쌍 파싱 후 병합, 여집합을 non_breathing으로 산출. DURATION_SEC=30.0 | 일치 | `1 parse_breathing_intervals.py:19,123-149` |
| 2 | 원본 오디오 16 kHz 리샘플 | TARGET_SR=16000, torchaudio resample. 후속 스크립트도 SR=16000 | 일치 | `2 extract_segments_from_json.py:22,47-48`; `3:35`; `4:17` |
| 3 | 각 세그먼트에 5개 라벨 부여 | breathing→diagnosis(Healthy/Crackle/Rhonchi/Wheezing), non→Non-breathing. 최종 5클래스 | 일치 | `2:24-25,82-114`; canonical 5클래스 `5:28` |
| 4 | 2초/0.5초 홉 변환 + **짧은 세그먼트 padding** | 2s/0.5s 슬라이딩 맞음. **그러나 padding 전혀 없음** — `if i1 > n: break`로 2초 미만 꼬리 조각을 버림. 스크립트 헤더에도 "패딩 제거" 명시. np.pad/tile/fix_length 등 어떤 형태도 부재 | **불일치** | `3 make_segments_2s_hop050.py:6("패딩 제거"),192-196`; grep 결과 패딩 코드 없음 |
| 5 | 각 윈도우 → 64-band log-Mel, per-sample 정규화 | N_MELS=64 melspectrogram→power_to_db, 시간축 평균(64,)→(v-μ)/σ per-sample 표준화 | 일치 | `4 extract_Logmel.py:18,25-36,63-64` |
| 6 | GroupShuffleSplit으로 참가자가 **train/val/test**에 나뉘지 않게 | 그룹 분할은 맞음(그룹키=파일명에서 시간범위 제거한 stem). **그러나 분할은 9:1 train/val 2세트만. test 세트 없음.** 그룹키에 `_1/_2` 세션 접미어가 포함되어, 그룹 단위 누수는 없으나 그룹 정의가 "참가자"가 아니라 "녹음/세션"임 | **부분일치** | `5:99-105`("9:1"), `5:30-38`(group=stem_without_time); 실행: train 21그룹/val 3그룹, 그룹 겹침 없음 |
| 7 | LR+MLP 앙상블 averaging | `P = 0.5*LR + 0.5*MLP` (동일 가중) | 일치 | `5:122` |
| 8 | 총 38명 환자가 학습에 사용 | **실제 학습 데이터: 24개 녹음 = 21명 환자.** RAW 폴더에 38개 wav 존재하나 14개(WEBSS 11개, KP002, KP012 2세션)는 호흡구간 주석 없어 features에 미포함. 38 wav=34명, 사용된 24 wav=21명. features_64mel.npz N=1368 윈도우 | **불일치** | 실행: npz 24그룹/21환자, metadata_windows.csv src_file=24, RAW=38(34환자); `01_inspect_npz.py` 출력 |
| 9 | 실시간 2초 윈도우가 **0.5초마다** boundary shifting 업데이트 | 앱 `HOP_S = 1.0` (1초 홉). 학습 세그먼트 생성만 0.5초. **tetho_softmax_export.py는 HOP_S=0.5**이나 주석은 "1s hop"이라 상충 | **불일치** | 앱 `Stethoscope_App_YH_V1.py:40`; export `tetho_softmax_export.py:29(0.5)+117/226(주석 1s)` |
| 10 | 계층적(먼저 breathing/non 분류 후 breathing만 4클래스) vs 플랫(5클래스 출력) | **플랫.** 단일 5클래스 확률 벡터를 한 번에 출력. 계층 분류 없음. non-breathing은 5클래스 중 한 클래스로 학습·추론 | 플랫이 확정 (L158-160 계층 서술과 불일치, L374-376 플랫 서술과 일치) | 학습 `5:28,115-122`; 앱 `_infer_one` 단일 5-way `App:134-150` |
| 11 | Fig 4B: 호흡 4클래스 정확도 ≥85%, non-breathing 65% | **배포 모델의 val 성능 재계산: 전체 accuracy 47.95%.** per-class recall: Crackle 0%, Healthy 0%, Wheezing 6.7%, Non-breathing 90.8%, Rhonchi=val 표본 0개(측정 불가). Fig 4B 수치와 심각하게 불일치. summary.json은 성능지표 미포함(카운트만) | **불일치** | 실행 `03_artifact_e2e.py`(배포 pkl 직접 사용); summary.json 전체=아래 |
| 12 | 알림 = (a)비정상 클래스 OR (b)HR/BR 임계 초과 시 트리거 | **HR/BR 계산 로직 없음.** card_hr/card_br는 "-- bpm" 정적 라벨. alert dot은 생성자에서 set_off만, set_on() 호출 없음. 비정상 클래스→알림 연결도 없음. 알림 로직 전무 | **불일치** | 앱 `:172-175,332-335,347-349`; grep: set_on 미호출, HR/BR 산출 부재 |
| 13 | 대표 사례 dominant class 확신도 >70% / 임계값 기반 진단 결정 | `get_latest_label_ui = argmax(probs - thresholds)` 존재하나 **반환값 폐기**(`_ = ...`). >70% 게이트 없음. UI는 원확률만 표시 | **부분일치/판단불가** | 앱 `:76-78,401`; grep: 0.7/70 게이트 없음 |
| 14 | thresholds.json class_names/thresholds, Rhonchi=Infinity 원인 | class_names=[Crackle,Healthy,Non-breathing,Rhonchi,Wheezing]. thresholds에서 **Rhonchi=Infinity**. 원인: val 분할(H001,KP007,KP017_1)에 **Rhonchi 표본 0개** → roc_curve가 양성 없이 thr에 inf 반환 → Youden argmax가 inf 선택. 재현 시 완전 동일 | 원인 확정 | `5:65-73`(compute_thresholds_ovr), 실행 재현 thr=inf, val Rhonchi n_pos=0 |
| 15 | 캡처 150-2000 Hz/센싱 200-1900 Hz 서술 vs 앱 FMIN=50 FMAX=7900 | 코드에 대역통과 필터(butter/bandpass 등) **전무**. 유일한 주파수 설정은 melspec FMIN=50/FMAX=7900. 150-2000/200-1900은 아날로그 프론트엔드 특성이지 코드가 강제하지 않음 | 부분일치(HW 특성이면 별개, 코드 강제는 없음) | grep: 필터 없음; `4:19`, App:`42` |
| 16 | BLE 4kHz 수신, 16kHz 리샘플, FMAX=7900의 의미 | BLE SR=4000(Nyquist 2000Hz)→16kHz 업샘플. **학습 RAW wav도 native 4000Hz**(38개 중 표본 30개 확인, 전부 4000). 따라서 64 mel band 중 **>2000Hz 29개(약 45%)는 학습·추론 양쪽 모두 리샘플 산물뿐**(실제 음향정보 없음). 내부적으로는 train/deploy 일관(도메인 불일치 없음)이나 FMAX=7900과 논문의 주파수 서술은 오해 소지 | 부분일치(일관성은 있으나 과학적 근거 취약) | 실행: RAW SR=4000, mel 45% band>2kHz; App:`29,42,480` |
| 17 | tetho_softmax_export.py 역할 | 학습과 동일 파라미터로 한 WAV 전체를 2s 슬라이딩 추론→CSV+그래프 저장하는 **오프라인 배치 추론/시각화 유틸**. 논문 Fig의 시간축 softmax 확률 곡선(대표 사례 트레이스) 생성에 대응. 앱과 파라미터 동일하나 HOP_S=0.5로 앱(1.0)과 다름 | 일치(용도)/불일치(홉) | `tetho_softmax_export.py:26-34,91-231` |

### summary.json 전문 (논문 사본)
```json
{
  "run_id": "run_20251107_194938",
  "n_total": 1368, "n_train": 1197, "n_val": 171,
  "classes": ["Crackle","Healthy","Non-breathing","Rhonchi","Wheezing"],
  "split": {"train": 0.9, "val": 0.1, "test": 0.0},
  "groups": {"train_unique": 21, "val_unique": 3}
}
```
→ 성능 지표(정확도/클래스별 recall) 없음. 카운트만. Fig 4B 수치를 뒷받침할 산출물이 저장물에 부재.

### 클래스 분포 (metadata_windows.csv = features와 동일 N=1368)
Non-breathing 718(52%), Wheezing 289, Crackle 149, Rhonchi 121, Healthy 91. → 심각한 불균형(Non-breathing 과반).

### 앱 버전 diff
`4 Realtime Pipeline_New/Stethoscope_App_YH_V1.py` vs `6 Final Deployment to Dr.Oh/…` = **주석 한→영 번역 및 창 타이틀 문구만 상이, 로직 완전 동일**(HOP_S=1.0 양쪽 동일). 논문 사본은 6번과 **완전 동일**(diff 없음).

## 추가 발견 (목록 외)

- **A. Rhonchi 영구 억제 버그 (치명적, 임상 안전).** thresholds Rhonchi=Infinity + 앱 `argmax(probs - thresholds)` 결합 시 Rhonchi의 `prob - inf = -inf`. 모델이 Rhonchi를 60% 확신해도 **Rhonchi는 진단 라벨로 절대 선택 불가**, 다른 클래스(시뮬레이션상 Crackle)로 오표시. UI에서 Rhonchi는 "Bronchi"로 표기되므로 사용자는 원인 파악 불가. 근거: `App:77` + 시뮬레이션(`_workspace/qa_scripts` 인라인).
- **B. val 성능 붕괴가 배포 모델 자체의 성능.** 재계산은 배포된 pkl로 직접 수행했고 결과가 저장 thresholds와 완전 일치 → 47.95% accuracy, 4개 breathing 중 3개(Crackle/Healthy/Wheezing) recall ≤7%가 실제 논문 모델의 성능. Fig 4B(≥85%)와 정면 배치.
- **C. val 표본 편중.** val 3그룹(H001, KP007, KP017_1)에 Rhonchi 0, 대부분 Non-breathing(87/171). test_size=0.10에 그룹 수 24개뿐이라 층화 불가 → 임계값·성능이 우연적 3그룹에 좌우됨.
- **D. 저장물에 성능·혼동행렬 이미지 없음.** 최종 스크립트에서 test/지표/이미지 저장 로직 제거됨(`5:3-6`). Fig 4B를 재현할 아티팩트가 파이프라인 산출물에 없음.

## 리뷰 대응 시 반드시 수정/해명 필요한 불일치 — 심각도 순

1. **[치명] Fig 4B 성능 수치 vs 실제 배포 모델(47.95% acc, breathing 3클래스 recall ≤7%).** 논문 그림과 코드 산출 모델이 동일 run임이 md5로 확정. Fig 4B가 별도 데이터/모델/집계 기준이면 그 출처·산출 스크립트를 명시해야 하며, 없다면 수치 재산출 또는 철회 필요. (담당: 학습/논문 저자)
2. **[치명] Rhonchi 임계값=Infinity → Rhonchi 영구 억제(발견 A) + Rhonchi가 학습·검증에서 사실상 평가 불가.** 임계값 산출 로직이 val에 양성 표본이 없을 때 inf를 그대로 배포. R2#6/R3#2 직결. 임계값 클램핑 또는 계층/stratified 재분할 필요. (담당: 학습)
3. **[상] "38명 환자" vs 실제 21명/24녹음.** R2#5(세트별 참가자/녹음 수) 직결. 참가자 수·세트 구성 재기술 필요. 14개 녹음 제외 사유(주석 부재)도 명시. (담당: 논문 저자)
4. **[상] train/val/test 3세트 서술 vs 실제 9:1 2세트(test 없음).** test 성능 주장 시 근거 소멸. R2#5 직결. (담당: 학습/저자)
5. **[상] "0.5초 업데이트" 서술 vs 앱 1초 홉(HOP_S=1.0).** 레이턴시(R2#8) 직결. export 스크립트는 0.5초라 내부 상충. 문서·코드 일원화 필요. (담당: 앱)
6. **[상] "padding applied" 서술 vs 실제 패딩 없음(2초 미만 폐기).** R2#9(패딩 인공패턴) 답변 전 서술 정정. (담당: 저자)
7. **[중] 계층적 분류(L158-160) vs 실제 플랫 5클래스.** 서술 통일 필요. (담당: 저자)
8. **[중] HR/BR 기반 알림·알림 트리거 로직 부재(L385-386).** 앱에 미구현. 구현하거나 서술에서 제거. (담당: 앱)
9. **[중] ">70% 확신도" 진단 게이트 부재(L394).** 앱은 확률만 표시, 임계 라벨 결과는 폐기. (담당: 앱)
10. **[중] 주파수 대역 서술(150-2000/200-1900) vs FMAX=7900 + BLE 4kHz.** mel 대역 약 45%가 2kHz 초과=정보 없음. 학습·추론 일관이라 성능엔 중립이나, 주파수 관련 서술과 FMAX 설정의 근거를 정리해야 함(나이퀴스트 상 2kHz 이상 무정보). (담당: 저자/신호처리)

## PASS로 남기는 정합성 항목 (문제 없음 명시)
- 아티팩트 end-to-end 추론: logmel 64차원 == scaler 입력 64, 확률벡터 길이 5, 합=1.0, NaN 없음. (PASS)
- 특성 파라미터 학습↔앱 일치: SR=16000, N_MELS=64, FMIN/FMAX=50/7900, WIN_MS/HOP_MS=64/32, WIN_S=2.0, per-sample 표준화 방식 모두 일치. (PASS) — 단 세그먼트 홉은 학습0.5/앱1.0로 설계상 상이(추론엔 무영향, 업데이트 주기 서술과만 충돌).
- 클래스 매핑: 학습 canonical=thresholds class_names=앱 UI_ORDER 집합 일치, 이름 기반 reorder 전 UI명 성립. class_names 순서=모델 확률 인덱스 정렬 확인. (PASS)
- 그룹 분할 누수: train/val 그룹 겹침 0, 세션접미어 제거 후 환자 겹침도 0. (PASS, 단 그룹정의=녹음/세션)
- 앙상블 가중치 0.5:0.5 학습·앱·export 모두 동일. (PASS)
- 앱 3버전(4New/6Deploy/논문사본) 로직 동일. (PASS)
