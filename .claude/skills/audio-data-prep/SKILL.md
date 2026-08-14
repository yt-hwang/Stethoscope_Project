---
name: audio-data-prep
description: 청진기 호흡음 데이터 준비 표준 절차. Excel 호흡 구간 라벨을 JSON으로 파싱, 오디오를 세그먼트로 추출, 64차원 로그멜 특성 추출까지의 전 과정. 새 녹음 추가, 라벨 파싱, 세그먼트/특성 추출, "데이터 다시 준비", "특성 재추출" 요청 시 반드시 이 스킬을 사용할 것.
---

# Audio Data Prep — 호흡음 데이터 준비 표준 절차

호흡음 원본(wav) + 라벨(Excel)을 학습 가능한 로그멜 특성으로 변환하는 표준 파이프라인.

## 특성 파라미터 규약 (Single Source of Truth)

이 파라미터는 실시간 앱 `RealtimeModelAdapter`(`Deployment/1 Final Pipeline/6 Final Deployment to Dr.Oh/Stethoscope_App_YH_V1.py`)와 동일해야 한다. **학습과 추론의 파라미터가 다르면 모델은 에러 없이 잘못된 확률을 출력한다** — 이것이 이 규약이 존재하는 이유다.

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| SR | 16000 Hz | 모든 오디오를 이 SR로 리샘플 |
| 세그먼트 윈도우 | 2.0 s | 학습·추론 공통 |
| 세그먼트 홉 | 0.5 s (학습) / 1.0 s (실시간) | 학습은 겹침으로 데이터 증강 |
| N_MELS | 64 | |
| FMIN / FMAX | 50 / 7900 Hz | 호흡음 주요 대역 |
| STFT WIN_MS / HOP_MS | 64 / 32 ms | |
| 채널 | mono | 스테레오는 평균으로 변환 |

파라미터 변경이 필요하면 학습 코드와 앱 어댑터를 **같은 커밋에서 동시에** 수정하고 qa-evaluator 검증을 거친다.

## 표준 절차

기존 스크립트를 재사용한다. 위치: `Deployment/1 Final Pipeline/5 Realtime Pipeline_Final/`

### Step 1: 라벨 파싱 (Excel → JSON)
- 스크립트: `1 parse_breathing_intervals.py`
- 입력 Excel 형식: 행1 = 파일명 + Inhale/Exhale 헤더, 행2 = 진단명 + 구간 시각(초)
- 출력 JSON 형식:
```json
{
  "KP001_WWS": {
    "diagnosis": "Bronchi",
    "breathing": [[0.124, 0.994], ...],
    "non_breathing": [[0.0, 0.124], ...]
  }
}
```
- 검증: 구간이 단조 증가하는지, 구간 끝이 오디오 길이 이내인지 확인. 위반 구간은 제외하고 리포트에 기록.

### Step 2: 구간 기반 세그먼트 추출
- 스크립트: `2 extract_segments_from_json.py` → `3 make_segments_2s_hop050.py`
- 호흡 구간의 88.8%가 2초 미만이므로, 짧은 구간은 0.5초 홉의 겹침 윈도우로 커버한다.
- 세그먼트 라벨: 윈도우가 걸친 구간의 클래스를 따르되, 경계에 걸친 모호 윈도우는 제외(라벨 노이즈 방지).

### Step 3: 로그멜 특성 추출
- 스크립트: `4 extract_Logmel.py`
- librosa melspectrogram → log 스케일 → per-sample 표준화
- 출력: 세그먼트당 64차원 × 시간프레임 특성 (플랫 벡터화 여부는 학습 스크립트 입력 형식에 맞춤)

### Step 4: 무결성 검증 및 리포트
`_workspace/01_data_report.md`에 기록:
- 처리 파일 수 / 제외 파일 수와 사유
- 세그먼트 수, 클래스별 분포
- 환자 ID 목록 (파일명 접두어 기준 — 이후 그룹 분할에 사용됨)
- 사용한 파라미터 전체 (규약과 다르면 굵게 표시)

## 주의사항

- 파일명 접두어(예: `KP001`)가 환자 ID다. 파일명을 변경하면 환자 그룹 분할이 깨지므로 원본 파일명을 보존한다.
- BLE 청진기 원본은 4kHz다. 학습 데이터와 도메인이 다를 수 있으므로, 실기기 녹음 데이터를 다룰 때는 리샘플 경로(4k→16k)를 리포트에 명시한다.
- Excel 라벨은 수작업 산출물이라 형식 편차가 있다. 파싱 실패 시 해당 행을 보고서에 남기고 사용자 확인을 받는다 — 추측으로 보정하지 않는다.
