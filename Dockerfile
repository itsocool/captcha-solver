FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# 런타임 설정은 Settings(pydantic-settings) 가 읽는다. 우선순위는
#   실제 환경 변수 > /app/.env > apps/web/core/config.py 의 필드 기본값
# 이라서 이미지에 ENV 기본값을 박으면 나중에 넣어준 .env 를 덮어써 버린다.
# 그래서 여기서는 아무 값도 고정하지 않는다:
#   - .env 를 제공하면(/app/.env 마운트 또는 compose env_file) 그 값이 쓰이고
#   - 없으면 config.py 의 기본값으로 뜬다.
# .env 는 .dockerignore 로 막아 이미지에 굽지 않는다.

# 의존성 먼저 설치해 레이어 캐시를 살린다 (uv.lock이 단일 소스)
# 워크스페이스 멤버(hypercaptcha)의 매니페스트도 있어야 lock 검증이 통과한다.
COPY pyproject.toml uv.lock README.md ./
COPY packages/python_3.12/hyperCaptcha/pyproject.toml packages/python_3.12/hyperCaptcha/README.md \
     ./packages/python_3.12/hyperCaptcha/
RUN uv sync --frozen --no-dev --no-install-workspace

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["fastapi", "run", "apps/web/app.py", "--host", "0.0.0.0", "--port", "8000"]
