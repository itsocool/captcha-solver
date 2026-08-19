#!/usr/bin/env python3
"""PostToolUse(Write|Edit) 훅: 편집한 소스 파일에 대응하는 docs/ 문서를 컨텍스트에 주입한다.

Claude Code 가 stdin 으로 넘기는 훅 입력(JSON)에서 file_path 를 꺼내 아래 MAPPING 과
대조하고, 걸리는 문서가 있으면 hookSpecificOutput.additionalContext 로 "이 문서들을
확인·갱신하라"는 리마인드를 돌려준다. 매칭이 없으면 아무것도 출력하지 않는다.

문서 ↔ 코드 대응 관계의 진실 소스는 이 파일의 MAPPING 이다. CLAUDE.md 의 표는 사람이
읽기 위한 요약이므로 여기를 고치면 CLAUDE.md 도 함께 맞춘다.
"""
import json
import os
import sys

# (경로 접두사 또는 정확한 상대 경로, [대응 문서...]) — 위에서부터 모두 검사해 합집합을 만든다.
# 경로는 저장소 루트 기준 상대 경로, 구분자는 '/'.
MAPPING = [
    # ── apps/web ─────────────────────────────────────────────────────────────
    ("apps/web/api/", ["docs/web-api-reference.md", "docs/web-architecture.md"]),
    ("apps/web/schemas/", ["docs/web-api-reference.md", "docs/web-domain-model.md"]),
    ("apps/web/services/", ["docs/web-architecture.md", "docs/web-domain-model.md"]),
    ("apps/web/core/db.py", ["docs/web-domain-model.md", "docs/web-architecture.md"]),
    ("apps/web/core/", ["docs/web-architecture.md", "docs/web-dev-guide.md"]),
    ("apps/web/templates/", ["docs/web-frontend-guide.md"]),
    ("apps/web/static/", ["docs/web-frontend-guide.md"]),
    ("apps/web/frontend/", ["docs/web-frontend-guide.md", "docs/web-architecture.md"]),
    ("apps/web/app.py", ["docs/web-architecture.md", "docs/web-dev-guide.md"]),
    ("apps/web/server.sh", ["docs/web-dev-guide.md"]),
    ("apps/web/server.ps1", ["docs/web-dev-guide.md"]),
    # ── DB 스키마 (apps/web 이 읽는 SQLite) ────────────────────────────────────
    ("db/", ["docs/web-domain-model.md"]),
    # ── hyperCaptcha 라이브러리 (학습/추론) ─────────────────────────────────────
    ("packages/python_3.12/hyperCaptcha/src/hypercaptcha/", ["docs/crnn_ctc.md", "AGENTS.md"]),
    ("packages/python_3.12/hyperCaptcha/pyproject.toml", ["docs/crnn_ctc.md"]),
    # ── 저장소 전반 / 배포 ─────────────────────────────────────────────────────
    ("pyproject.toml", ["docs/web-dev-guide.md", "docs/codebase-analysis.md"]),
    ("Dockerfile", ["docs/web-dev-guide.md", "docs/codebase-analysis.md"]),
    ("docker-compose", ["docs/web-dev-guide.md", "docs/codebase-analysis.md"]),
    ("apps/cli/", ["docs/codebase-analysis.md"]),
    ("apps/springBoot/", ["docs/codebase-analysis.md"]),
]

# 이 접두사에 걸리면 리마인드하지 않는다 (문서 자체, 설정, 캐시 등).
IGNORE_PREFIXES = ("docs/", ".claude/", "captcha_data/", "__pycache__", ".dev/", "tests/")


def to_relative(path: str) -> str:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    path = os.path.abspath(path)
    root = os.path.abspath(root)
    try:
        rel = os.path.relpath(path, root)
    except ValueError:  # 다른 드라이브 등
        return path.replace(os.sep, "/")
    return rel.replace(os.sep, "/")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path") or (payload.get("tool_response") or {}).get("filePath")
    if not file_path:
        return 0

    rel = to_relative(file_path)
    if rel.startswith("..") or any(seg in rel for seg in IGNORE_PREFIXES):
        return 0

    docs: list[str] = []
    for prefix, targets in MAPPING:
        if rel.startswith(prefix) or ("/" not in prefix and os.path.basename(rel).startswith(prefix)):
            for d in targets:
                if d not in docs:
                    docs.append(d)

    if not docs:
        return 0

    context = (
        f"[docs-sync] `{rel}` 을(를) 수정했습니다. 이 파일은 다음 문서의 대상 코드입니다: "
        + ", ".join(f"`{d}`" for d in docs)
        + ". 변경이 API 계약·데이터 구조(DTO/엔티티/dict 형태)·학습/추론 동작·설정값·실행 절차 중 "
        "하나라도 바꿨다면, 같은 작업 안에서 해당 문서의 관련 절을 찾아 갱신하세요. "
        "문서에 영향이 없는 변경(리팩터링, 주석, 포맷)이면 넘어가도 됩니다."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        },
        "suppressOutput": True,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
