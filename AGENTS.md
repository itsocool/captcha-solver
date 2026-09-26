# 기본 언어

- 응답, 코드 주석, 문서는 기본적으로 한국어로 작성한다. 사용자가 다른 언어를 명시적으로 요청하면 해당 언어를 사용한다.
- 군더더기 없이 간결하고 전문용어, 약어를 활용하여 정확하게 응답 한다.

# 요구사항 기반 작업 하네스

- 기능 구현·수정·검증 및 설계 문서 작업의 기준 문서는 [요구사항 정의서](docs/00.requirements.md)다. 작업 시작 시 관련 요구사항 ID와 수용 기준을 읽고, 대상 앱의 `AGENTS.md` 및 구현 근거를 확인한다.
- 정의서의 **확정**, **현행 기준**, **미정**을 구분한다. 미정 사항을 임의로 확정하거나 화면·문서의 존재만으로 기능이 완성됐다고 판단하지 않는다.
- 현재 사용자 요청이 기존 문서와 다르면 사용자 요청을 우선하며, 영향을 받는 요구사항과 수용 기준을 함께 갱신한다. 미정 사항이 작업 결과를 바꾸는 경우에만 필요한 정보를 확인한다.
- 변경 후 정의서의 검증 표에서 영향 범위에 맞는 검사를 수행한다. 완료 보고에는 대상 ID, 변경 사항, 실제 검증 결과와 미검증·차단 항목을 간결하게 남긴다.
- 상세 요구사항과 실행 명령은 정의서에서 관리한다. 이 파일에는 참조·실행 규칙만 유지하고 비밀번호나 API Key를 기록하지 않는다.
- 공용 코드 변경은 [패키지 경계 규칙](packages/README.md)을 따른다.

# Git/GitHub 저장소 작업

- Git/GitHub 저장소 작업은 전용 서브에이전트를 생성하여 위임한다. 모델은 `GPT-6-Luna`, 추론 수준은 `medium`으로 지정한다.
- 서브에이전트 생성 시 `fork_turns="none"`으로 설정하고, 작업 범위, 저장소 경로, 필요한 맥락과 사용자 승인 범위를 명시적으로 전달한다.
- 전용 서브에이전트는 위임받은 작업을 직접 처리하고, 변경 사항과 검증 결과를 주 에이전트에 보고한다. 동일 작업을 다시 위임하지 않는다.
- 위임은 사용자 승인 범위를 확대하지 않는다. 사용자 변경 사항을 보존하고, 파괴적인 Git 작업은 명시적인 승인 없이 수행하지 않는다.

# Simple 에이전트

- 간단한 질문에 대한 답변, 이름 변경, 오타 수정 등 범위가 명확한 단순 작업은 `simple` 전용 서브에이전트를 생성하여 위임한다. 모델은 `GPT-6-Luna`, 추론 수준은 `medium`으로 지정한다.
- 생성 시 `fork_turns="none"`으로 설정하고, 작업 범위, 필요한 맥락과 대상 경로, 사용자 승인 범위를 전달한다.
- `simple` 에이전트는 작업을 직접 처리하고 필요한 검증 후 결과를 주 에이전트에 보고한다. 동일 작업을 다시 위임하지 않으며, 사용자 변경 사항과 승인 범위를 보존한다.
- Git/GitHub 저장소 작업은 해당 전용 서브에이전트 규칙을 우선 적용한다.

# Serena

Use Serena as the default tool for semantic code navigation and precise edits.

Prefer Serena for:

- finding classes, functions, methods, interfaces, and constants
- inspecting symbol definitions
- finding references to symbols
- understanding nearby code structure
- editing or renaming specific symbols

Prefer symbol-level operations such as:

```
get_symbols_overview
find_symbol
find_referencing_symbols
replace_symbol_body
insert_before_symbol
insert_after_symbol
```

Avoid reading entire source files when symbol-level access is sufficient.

Use this priority:

```
Serena symbol lookup
→ Serena references
→ targeted text search
→ partial file read
→ full file read only when necessary
```

Retrieve only the code required for the current task to minimize context and token usage.
