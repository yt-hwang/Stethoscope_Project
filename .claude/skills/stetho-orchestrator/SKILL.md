---
name: stetho-orchestrator
description: 청진기 호흡음 ML 파이프라인 에이전트 팀을 조율하는 오케스트레이터. 데이터 준비→모델 학습→QA 검증→실시간 앱 배포의 전체 또는 일부를 수행. 새 데이터로 모델 갱신, 모델 재학습, 앱 배포, 파이프라인 실행/재실행/업데이트/수정/보완, "학습만 다시", "배포만 다시", 이전 결과 개선, 성능 개선 실험, 배포 패키지 갱신 등 호흡음 파이프라인 관련 작업 요청 시 반드시 이 스킬을 사용할 것. 단순 코드 질문은 직접 응답 가능.
---

# Stetho Orchestrator — 호흡음 ML 파이프라인 조율

호흡음 데이터 준비부터 실시간 앱 배포까지의 파이프라인을 에이전트 팀으로 실행하는 통합 스킬.

## 실행 모드: 에이전트 팀 (기본)

단일 단계만 필요한 소규모 요청(예: "특성만 재추출")은 해당 에이전트 1명을 서브 에이전트(Agent 도구, `model: "opus"`)로 직접 호출해도 된다. 2개 이상 단계가 얽히면 팀을 구성한다.

## 에이전트 구성

| 팀원 | 에이전트 타입 | 역할 | 스킬 | 출력 (`_workspace/`) |
|------|-------------|------|------|---------------------|
| data-engineer | general-purpose | 라벨 파싱, 세그먼트/특성 추출 | audio-data-prep | `01_data_*` |
| ml-trainer | general-purpose | 앙상블 학습, 임계값 최적화 | model-training | `02_train_*` |
| deployment-engineer | general-purpose | 앱 수정, 아티팩트 교체, 패키징 | realtime-deployment | `03_deploy_*` |
| qa-evaluator | general-purpose | 경계면 정합성 검증 (incremental) | pipeline-qa | `04_qa_report.md` |

에이전트 상세 정의: `.claude/agents/{이름}.md` — 팀원 스폰 프롬프트에 해당 정의 파일과 스킬을 읽도록 지시한다. 모든 스폰에 `model: "opus"` 명시.

## 워크플로우

### Phase 0: 컨텍스트 확인 (후속 작업 지원)

1. 프로젝트 루트의 `_workspace/` 존재 여부 확인
2. 실행 모드 결정:
   - **미존재** → 초기 실행, Phase 1로
   - **존재 + 부분 수정 요청** (예: "임계값만 다시", "앱만 고쳐줘") → **부분 재실행**: 해당 에이전트만 호출, 이전 산출물 경로를 프롬프트에 포함해 기존 결과를 읽고 반영하게 함. 하류 산출물이 영향받으면(예: 재학습 → 배포 갱신) 영향 범위를 사용자에게 알린다.
   - **존재 + 새 입력 제공** (새 데이터셋 등) → 기존 `_workspace/`를 `_workspace_{YYYYMMDD_HHMMSS}/`로 이동 후 초기 실행
3. 요청 범위 판별: 전체 파이프라인 / 데이터만 / 학습만 / 배포만 / 검증만

### Phase 1: 준비
1. 입력 확인 — 대상 오디오/라벨 경로, 실험 조건, 배포 대상
2. `_workspace/` 생성 (프로젝트 루트 기준)

### Phase 2: 팀 구성

요청 범위에 필요한 팀원만 스폰한다. qa-evaluator는 2단계 이상 실행 시 항상 포함한다.

```
TeamCreate(team_name: "stetho-team", members: [
  { name: "data-engineer", agent_type: "general-purpose", model: "opus",
    prompt: ".claude/agents/data-engineer.md와 .claude/skills/audio-data-prep/SKILL.md를 읽고 역할 수행. {구체 작업}" },
  { name: "ml-trainer", ... },
  { name: "deployment-engineer", ... },
  { name: "qa-evaluator", agent_type: "general-purpose", model: "opus",
    prompt: ".claude/agents/qa-evaluator.md와 .claude/skills/pipeline-qa/SKILL.md를 읽고 incremental QA 수행. {검증 범위}" },
])
```

작업 등록 (의존성 명시):
```
TaskCreate(tasks: [
  { title: "데이터 준비", assignee: "data-engineer" },
  { title: "데이터 QA (QA-6)", assignee: "qa-evaluator", depends_on: ["데이터 준비"] },
  { title: "모델 학습", assignee: "ml-trainer", depends_on: ["데이터 QA (QA-6)"] },
  { title: "학습 QA (QA-3,4,5)", assignee: "qa-evaluator", depends_on: ["모델 학습"] },
  { title: "배포 반영", assignee: "deployment-engineer", depends_on: ["학습 QA (QA-3,4,5)"] },
  { title: "배포 QA (QA-1,2,4)", assignee: "qa-evaluator", depends_on: ["배포 반영"] },
])
```

### Phase 3: 파이프라인 실행 (팀 자체 조율)

팀원 간 통신 규칙:
- data-engineer → ml-trainer: 특성 경로 + shape (SendMessage)
- ml-trainer → deployment-engineer: 아티팩트 세트 경로
- 각 팀원 → qa-evaluator: 단계 완료 직후 검증 요청 (**incremental QA — 전체 완성 후 몰아서 검증 금지**)
- qa-evaluator → 담당 팀원: FAIL 시 재현 방법과 함께 즉시 전달, 차단급(QA-1~3)은 리더에게도 보고

리더는 TaskGet으로 진행 모니터링. **qa-evaluator가 차단급 FAIL을 보고하면 하류 작업을 중단**시키고 담당 팀원의 수정 완료 후 재개한다.

### Phase 4: 통합 및 완료 확인
1. 전 작업 완료 대기 (TaskGet)
2. `_workspace/04_qa_report.md` 확인 — 차단급 항목이 모두 PASS인지
3. 최종 요약 작성: 데이터 통계, 모델 성능(이전 대비), QA 결과, 배포 패키지 위치

### Phase 5: 정리
1. 팀원 종료 요청 (SendMessage) → TeamDelete
2. `_workspace/` 보존 (감사 추적용)
3. 사용자에게 결과 요약 보고 + 개선 피드백 기회 제공 ("결과에서 개선할 부분이 있나요?")

## 데이터 흐름

```
[data-engineer] → 01_data_* → [ml-trainer] → 02_train_artifacts/ → [deployment-engineer] → 03_deploy_package/
        │                          │                                       │
        └────── 검증 요청 ──────────┴──────────── 검증 요청 ────────────────┘
                                   ↓
                            [qa-evaluator] → 04_qa_report.md (incremental)
                                   ↓ FAIL 시
                            담당 팀원에게 재작업 요청
```

## 에러 핸들링

| 상황 | 전략 |
|------|------|
| 팀원 1명 실패/중지 | 리더가 감지 → SendMessage로 상태 확인 → 재시작, 재실패 시 해당 산출물 누락을 명시하고 진행 여부를 사용자에게 확인 |
| QA 차단급 FAIL | 하류 작업 중단 → 담당 팀원 수정 → 해당 QA 항목만 재검증 후 재개 |
| QA 환경 문제로 검증 불가 | 해당 항목 SKIP 처리(사유 기록), PASS로 위장 금지. 최종 보고에 명시 |
| 라이브러리/환경 오류 | 1회 설치 시도, 실패 시 필요한 수동 조치를 사용자에게 안내 |
| 데이터 상충 (라벨 vs 오디오) | 삭제하지 않고 출처 병기, 리포트에 기록 |

## 테스트 시나리오

### 정상 흐름: 새 데이터로 모델 갱신 후 배포
1. 사용자: "Audio shared에 새 녹음 추가했어. 모델 갱신하고 배포 패키지 만들어줘"
2. Phase 0: `_workspace/` 없음 → 초기 실행, 범위 = 전체
3. Phase 2: 4명 팀 + 6개 작업 등록
4. Phase 3: 데이터 준비 → QA-6 PASS → 학습 → QA-3/4/5 PASS → 배포 → QA-1/2/4 PASS
5. 예상 결과: 새 버전 배포 폴더 + `_workspace/` 산출물 일체 + 성능 비교 요약

### 에러 흐름: 특성 파라미터 불일치 발견
1. Phase 3에서 qa-evaluator가 QA-1 FAIL 보고 (학습 N_MELS=64 vs 앱 수정본 N_MELS=128)
2. 리더가 배포 작업 중단, deployment-engineer에게 수정 지시
3. 수정 후 qa-evaluator가 QA-1만 재검증 → PASS → 배포 재개
4. 최종 보고서에 FAIL 이력과 수정 내역 명시

### 부분 재실행: 임계값만 조정
1. 사용자: "Wheezing 재현율이 낮아. 임계값만 다시 잡아줘"
2. Phase 0: `_workspace/` 존재 + 부분 수정 → ml-trainer만 서브 에이전트로 호출 (팀 불필요)
3. ml-trainer가 기존 `02_train_report.md`를 읽고 임계값 재최적화 → qa-evaluator 서브 호출로 QA-2/4 검증
4. 사용자에게 배포본 갱신 필요 여부 확인
