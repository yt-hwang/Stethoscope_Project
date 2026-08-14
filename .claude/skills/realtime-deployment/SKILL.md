---
name: realtime-deployment
description: 실시간 청진기 앱 수정 및 배포 패키징 표준 절차. PyQt5 GUI 앱(Stethoscope_App_YH_V1.py) 수정, BLE 스트림 처리, 모델 아티팩트 교체, Dr.Oh 전달용 배포 패키지 생성. 앱 수정, 새 모델 배포, 배포 패키지 생성/업데이트, "앱에 반영" 요청 시 반드시 이 스킬을 사용할 것.
---

# Realtime Deployment — 실시간 앱 배포 표준 절차

학습된 모델을 실시간 앱에 통합하고 외부 전달용 패키지를 만드는 절차.

## 앱 구조 (기준: `Deployment/1 Final Pipeline/6 Final Deployment to Dr.Oh/Stethoscope_App_YH_V1.py`)

```
BLE 스트림 (4kHz, 180바이트 패킷, UUID 0000eef2-...)
  → 16kHz 리샘플 → 2초 버퍼 누적 → 1초 홉마다 추론
  → RealtimeModelAdapter: 64-dim 로그멜 → scaler → 0.5·LR + 0.5·MLP
  → UI: 확률 테이블 + 실시간 그래프 + argmax(확률−임계값) 진단 라벨
```

핵심 상수 (학습 파이프라인과 일치 필수 — `audio-data-prep` 스킬의 규약 표 참조):
`SR=16000, WIN_S=2.0, HOP_S=1.0, N_MELS=64, FMIN=50, FMAX=7900, WIN_MS=64, HOP_MS=32`
`UI_ORDER = ['Healthy', 'Crackle', 'Rhonchi', 'Wheezing', 'Non-breathing']`

## 모델 교체 절차

1. ml-trainer의 아티팩트 4종(`scaler.pkl`, `model_lr.pkl`, `model_mlp.pkl`, `thresholds.json`) 완전성 확인
2. 앱의 모델 디렉토리에 복사 — 기존 모델은 `models_backup_{날짜}/`로 보관
3. **스모크 테스트 실행** (GUI 없이):
   - 앱 모듈 import 성공
   - 아티팩트 4종 로드 성공
   - 2초 더미 오디오(np.random 또는 실제 wav 조각)로 어댑터 추론 → 5개 클래스 확률 합 ≈ 1.0 확인
4. thresholds.json 키와 UI_ORDER 매핑 확인
5. qa-evaluator에게 경계면 검증 요청 후 통과 시 완료

## 배포 패키지 구성

새 배포는 기존 `6 Final Deployment to Dr.Oh/`를 덮어쓰지 않고 새 버전 폴더로 만든다 (예: `7 Deployment vN_{날짜}/`).

패키지 내용물:
```
{배포폴더}/
├── Stethoscope_App_YH_V1.py     # 앱 (버전 주석 갱신)
├── {model_dir}/                  # 아티팩트 4종
├── requirements.txt              # PyQt5, librosa, scikit-learn, bleak 등
└── README.md                     # 실행 방법, 모델 버전, 변경 내역
```

zip 전달이 필요하면 폴더 완성 후 압축한다.

## 앱 수정 시 주의사항

- **BLE 수신부(4kHz)와 리샘플 경로를 건드리지 않는다.** 하드웨어 프로토콜에 묶여 있다.
- 추론 어댑터 파라미터 변경은 단독으로 하지 않는다 — 학습 파이프라인과 동시 변경 + qa-evaluator 검증이 원칙이다. 한쪽만 바꾸면 에러 없이 확률만 조용히 틀어진다.
- GUI 스레드에서 추론을 직접 돌리지 않는다(기존 구조의 워커 스레드 유지) — 블로킹되면 BLE 패킷이 유실된다.
- 수정 후에는 반드시 스모크 테스트(위 3번)를 재실행한다.

## 산출물

- `_workspace/03_deploy_package/` 또는 새 버전 폴더
- `_workspace/03_deploy_report.md` — 변경 내역, 모델 버전(학습 리포트 링크), 스모크 테스트 결과
