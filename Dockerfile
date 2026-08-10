FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

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
