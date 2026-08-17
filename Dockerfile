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
# --mount=type=cache: uv 휠 캐시가 레이어에 구워지면 그만큼 이미지가 커진다.
# 빌드 캐시로 빼면 재빌드는 빠르면서 이미지에는 남지 않는다.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace

# COPY . . 대신 실행에 필요한 것만 복사한다. .dockerignore 가 images/·*.sqlite3·.env 를
# 이미 걸러내지만, 여기서 대상을 좁혀 불필요한 파일이 이미지에 들어가지 않게 한다.
#   apps/web                     : FastAPI 앱(라우터·서비스·templates·static·favicon)
#   packages/.../hyperCaptcha/src: 워크스페이스 패키지(추론·전처리) 소스 (매니페스트는 위에서 복사)
#   db                           : schema.sql·seed (기동 시 init_db 가 읽는다)
#   captcha_data                 : 모델(onnx/ort/pth/meta). compose 는 볼륨으로 덮어쓴다.
COPY apps/web ./apps/web
COPY packages/python_3.12/hyperCaptcha/src ./packages/python_3.12/hyperCaptcha/src
COPY db ./db
COPY captcha_data ./captcha_data
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 개발 서버(fastapi dev): 자동 리로드·상세 오류. 컨테이너 밖에서 붙으려면 0.0.0.0.
# 코드 변경을 실시간 반영하려면 compose 에서 소스를 볼륨 마운트해야 한다(주석 참고).
CMD ["fastapi", "dev", "apps/web/app.py", "--host", "0.0.0.0", "--port", "8000"]
