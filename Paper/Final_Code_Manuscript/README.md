# 논문 최종 사용 코드 (Manuscript Version)

**논문:** "A Soft Wearable System for Real-time Respiratory Disease Diagnosis and Vibrotactile Intervention" (Science Advances 투고, Oh, Hwang et al.)
**분리 일자:** 2026-08-13
**원본 위치:** 아래 표 참조 (이 폴더는 사본 — 원본은 그대로 보존됨)

## 이 버전이 "최종"인 근거

1. **아티팩트 해시 일치**: `5 Realtime Pipeline_Final/model/run_20251107_194938/`의 4개 파일(md5)이 Dr.Oh에게 배포된 `6 Final Deployment to Dr.Oh/run_20251107_194938/`와 **완전 동일** → 이 학습 실행이 곧 배포·논문 모델.
2. **앱이 이 모델을 로드**: `Stethoscope_App_YH_V1.py` L23의 `MODEL_DIR`이 `5 Realtime Pipeline_Final/model/run_20251107_194938`을 가리킴.
3. **버전 계보**: `2 Model Training with Replayed Sound`의 동명 스크립트들은 이전 버전(test 세트 포함, 경로 다름). 5번 폴더 스크립트가 "(변경 요약)" 주석과 함께 최종 수정본.
4. **논문 서술과 일치**: 16kHz 리샘플, 2초 윈도우/0.5초 홉 세그먼트, 64-band log-Mel per-sample 정규화, GroupShuffleSplit(환자 그룹, 9:1), LR+MLP 앙상블(0.5:0.5), OVR ROC 임계값 — Manuscript L342-360 및 Fig 4A와 대응.

## 폴더 구성

| 폴더 | 내용 | 원본 위치 |
|------|------|----------|
| `1_training_pipeline/` | 학습 파이프라인 스크립트 1~5 + 특성 파일(features_64mel.npz) + summary.json | `Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/` |
| `2_realtime_app/` | 실시간 PyQt5 앱 + softmax export 스크립트 | `Deployment/1 Final Pipeline/6 Final Deployment to Dr.Oh/` |
| `3_model_run_20251107_194938/` | 배포 아티팩트 4종 (scaler, LR, MLP, thresholds) | 위 두 폴더에 동일 사본 존재 (md5 검증됨) |

## 파이프라인 흐름 (논문 Fig 4A 대응)

```
Excel 호흡구간 라벨 → (1) parse → breathing/non-breathing JSON
→ (2) 구간 기반 세그먼트 추출 → (3) 2s 윈도우/0.5s 홉 + coverage 기반 라벨링 (5클래스)
→ (4) 64-band log-Mel, per-sample 표준화 → features_64mel.npz
→ (5) GroupShuffleSplit(환자 그룹, 9:1) → StandardScaler → LR + MLP → 0.5·LR+0.5·MLP 앙상블
     → OVR ROC 임계값(thresholds.json) → run_20251107_194938 아티팩트
→ 앱: BLE 4kHz 수신 → 16kHz 리샘플 → 2s 버퍼 → 추론 → UI 표시
```

## ⚠️ 논문 서술 vs 코드 불일치 — 전수 감사 결과 (2026-08-13, 상세: `_workspace/04_qa_paper_code_audit.md`)

QA 에이전트가 코드 정독 + 실행 검증(npz/pkl 로드, 분할 재현, 성능 재계산)으로 확정한 목록. 심각도 순:

1. **[치명] Fig 4B(≥85%)의 출처가 프로젝트 내에 없음.** 최종 파이프라인은 성능 지표를 저장하지 않으며(summary.json엔 카운트만), 배포 모델로 val 재계산 시 47.95% (단, val은 3그룹/171윈도우의 퇴화 세트라 일반 성능 지표로는 부적합). 프로젝트 전체의 confusion matrix 산출물(2번 폴더 63.2%, Backup Plan 87.7%, Group_Split 79.7%) 중 Fig 4B(89/85/87/94/65)와 일치하는 것 없음. → **1저자(오새웅)에게 Fig 4B 산출 데이터/스크립트 확인 필수.**
2. **[치명] Rhonchi 임계값 = Infinity → 앱에서 Rhonchi 진단 영구 억제.** 원인 확정: val 3그룹에 Rhonchi 표본 0개 → roc_curve가 inf 반환 → 그대로 배포. 앱의 `argmax(probs - thresholds)`에서 Rhonchi는 -inf가 되어 절대 선택 불가.
3. **[상] "38명 환자 학습" vs 실제 21환자/24녹음.** RAW 38개 wav 중 14개(WEBSS 11개 등)는 호흡 구간 주석이 없어 특성 추출에서 제외됨. 최종 N=1368 윈도우.
4. **[상] "train/validation/test" vs 실제 9:1 train/val 2세트** (test 로직은 이전 버전인 2번 폴더에만 존재).
5. **[상] "0.5초마다 업데이트" vs 배포 앱 `HOP_S=1.0`.** tetho_softmax_export.py는 0.5로 자체 상충.
6. **[상] "짧은 세그먼트 padding" vs 실제 패딩 없음** — 2초 미만 꼬리는 폐기(`3 make_segments:192-196`, 헤더에 "패딩 제거" 명시).
7. **[중] 계층적 분류 서술(L158-160) vs 실제 플랫 5-클래스** (L374-376 서술이 코드와 일치).
8. **[중] HR/BR 기반 알림(L385-386) 미구현** — 앱에 HR/BR 계산·알림 트리거 로직 없음 (카드 "-- bpm" 정적).
9. **[중] ">70% 확신도" 게이트 없음** — 앱은 원확률만 표시, 임계값 진단 결과는 계산 후 폐기(`_=`).
10. **[중] 주파수 서술 정리 필요** — BLE 원본 4kHz(나이퀴스트 2kHz), RAW 학습 wav도 native 4kHz인데 melspec FMAX=7900 → 64밴드 중 ~45%는 무정보(학습·추론 동일 조건이라 성능엔 중립).

**정합성 PASS 항목**: 특성 파라미터 학습↔앱 완전 일치, 클래스 이름 매핑/재정렬 성립, 그룹 누수 0, 앙상블 0.5:0.5 일관, 아티팩트 E2E 추론 정상, 앱 3개 버전(4번/6번/사본) 로직 동일.
