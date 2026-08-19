# CLAUDE.md

@AGENTS.md

## 문서 참조 규칙 (`docs/`)

`docs/` 는 이 저장소의 **설계·계약 레퍼런스**다. 코드를 읽기 전에 관련 문서부터 확인하면 탐색 비용이 크게 준다. 작업 유형별로 먼저 볼 문서:

| 작업 | 먼저 읽을 문서 | 대상 코드 |
|---|---|---|
| 웹 API 엔드포인트 추가/수정, 요청·응답 형식 | `docs/web-api-reference.md` | `apps/web/api/`, `apps/web/schemas/` |
| 웹 레이어 구조, 요청 흐름, 동시성(SSE·워커 스레드) | `docs/web-architecture.md` | `apps/web/app.py`, `apps/web/services/`, `apps/web/core/` |
| DTO / 도메인 객체 / SQLite 엔티티, 데이터 흐름 | `docs/web-domain-model.md` | `apps/web/schemas/`, `apps/web/services/`, `apps/web/core/db.py`, `db/*.sql` |
| 로컬 실행, 환경 변수, 테스트, 확장 절차 | `docs/web-dev-guide.md` | `apps/web/server.*`, `pyproject.toml`, `Dockerfile*`, `docker-compose*` |
| 템플릿 / 정적 JS / 프런트 구조 | `docs/web-frontend-guide.md` | `apps/web/templates/`, `apps/web/static/`, `apps/web/frontend/` |
| CRNN 모델, 학습·추론 파이프라인, 저장 포맷, CLI | `docs/crnn_ctc.md` | `packages/python_3.12/hyperCaptcha/src/hypercaptcha/` |
| 저장소 전체 개요 (Rust CLI, Spring Boot 포함) | `docs/codebase-analysis.md` | `apps/cli/`, `apps/springBoot/`, 루트 |

- 문서와 코드가 다르면 **코드가 진실**이다. 그 자리에서 문서를 고치거나, 못 고치면 불일치를 사용자에게 알린다.
- `docs/superpowers/` 는 과거 설계 스펙/플랜 기록이다. 참고용일 뿐 현재 동작의 근거로 삼지 않는다.

## 문서 갱신 규칙

위 표의 "대상 코드"를 수정했고 그 변경이 아래 중 하나에 해당하면 **같은 작업 안에서** 대응 문서의 관련 절을 갱신한다:

- API 계약 (경로, 파라미터, 응답 필드, 오류 코드, SSE 이벤트 형태)
- 데이터 구조 (Pydantic 모델, 서비스 dict 키, SQLite 컬럼/제약, 파일시스템 레이아웃)
- 학습/추론 동작 (모델 구조, 손실 함수, 스케줄러, 증강, 디코딩, 저장 포맷, `on_event` 프로토콜)
- 설정값·기본값 (환경 변수, `PARAM_SPEC`, 레지스트리의 캡차 종류)
- 실행 절차 (기동 명령, 스크립트, Docker)

리팩터링·주석·포맷 변경처럼 문서 내용에 영향이 없으면 건너뛴다.

`.claude/hooks/docs-reminder.py` 가 `Write|Edit` 직후 대응 문서 목록을 컨텍스트에 자동 주입한다(`.claude/settings.json` 의 PostToolUse 훅). 파일 ↔ 문서 대응의 진실 소스는 그 스크립트의 `MAPPING` 이며, 문서를 추가하거나 코드 위치를 옮기면 스크립트와 위 표를 함께 맞춘다.
