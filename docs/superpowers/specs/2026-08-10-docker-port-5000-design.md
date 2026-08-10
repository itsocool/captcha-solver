# Docker 웹 포트 5000 통일 설계

## 목표

Docker로 실행하는 FastAPI 서비스의 호스트 포트, 컨테이너 포트, 실행 포트, 헬스체크 포트를 모두 `5000`으로 통일한다. 웹 실행 설정은 루트 `.env`의 `WEB_HOST`, `WEB_PORT`, `WEB_DEBUG`를 따른다.

## 변경 범위

- `Dockerfile`의 노출 포트와 헬스체크 URL을 `5000`으로 변경한다.
- `Dockerfile`의 실행 명령을 `python apps/web/app.py`로 변경한다. 이 진입점은 Pydantic Settings를 통해 `.env` 값을 읽어 Uvicorn에 전달한다.
- `docker-compose.yml`의 포트 매핑을 `5000:5000`으로 변경한다.
- `README.md`와 `AGENTS.md`의 Docker 및 FastAPI 포트 안내를 `5000`으로 정정한다.

## 설정 흐름

Compose가 루트 `.env`를 컨테이너 환경 변수로 주입하고, `apps/web/core/config.py`가 이를 읽는다. `apps/web/app.py`는 읽은 호스트, 포트, 디버그 값을 `uvicorn.run`에 전달한다. `.env`가 없으면 코드 기본값인 `0.0.0.0:5000`과 `WEB_DEBUG=false`를 사용한다.

## 오류 처리와 호환성

기존 `/health` 엔드포인트와 Compose의 `captcha_data` 마운트는 변경하지 않는다. 포트를 한 값으로 통일하므로 별도 셸 명령이나 환경 변수 보간은 추가하지 않는다.

## 검증

- Dockerfile과 Compose 설정 렌더링이 유효한지 확인한다.
- 이미지 빌드 후 컨테이너가 `5000`에서 기동하는지 확인한다.
- `http://localhost:5000/health`가 HTTP 응답을 반환하는지 확인한다.
- 저장소에 남은 Docker/FastAPI용 `8000` 및 `5001` 안내가 없는지 검색한다.
