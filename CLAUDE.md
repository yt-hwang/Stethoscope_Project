# Stethoscope Project

청진기 호흡음 분류 ML 프로젝트 — 호흡음(Healthy/Crackle/Rhonchi/Wheezing/Non-breathing) 분류 모델을 학습하고 PyQt5 실시간 앱으로 배포한다.

이 프로젝트는 Science Advances 투고 논문 "A Soft Wearable System for Real-time Respiratory Disease Diagnosis and Vibrotactile Intervention"의 ML 파이프라인을 담당한다. 논문 PDF/SI/리뷰 리포트와 논문에 실제 사용된 최종 코드 사본은 `Paper/` 폴더에 있다 (`Paper/Final_Code_Manuscript/README.md`에 최종 버전 근거와 논문-코드 불일치 목록 정리됨). 현재 과제: 리뷰 코멘트 중 ML 관련 항목 대응.

## 하네스: 호흡음 ML 파이프라인

**목표:** 데이터 준비 → 앙상블 학습 → 정합성 QA → 실시간 앱 배포 파이프라인의 자동화 및 품질 보장

**트리거:** 호흡음 데이터 처리, 모델 학습/재학습, 검증, 실시간 앱 수정/배포 관련 작업 요청 시 `stetho-orchestrator` 스킬을 사용하라. 단순 코드 질문은 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-08-13 | 초기 구성 (에이전트 4 + 스킬 5) | 전체 | revfactory/harness 기반 하네스 구축 |
| 2026-08-13 | 클래스 "순서 일치"→"이름 집합+class_names 정렬" 검증으로 정정, 트리거 문구 보강 | skills/model-training, skills/pipeline-qa, agents/qa-evaluator | 드라이런 검증에서 실제 코드는 이름 기반 재정렬 방식임을 확인 (거짓 FAIL 방지) |
