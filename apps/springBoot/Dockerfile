# syntax=docker/dockerfile:1
FROM eclipse-temurin:25-jdk AS build
WORKDIR /build

# 이미지엔 Maven 이 없으니 설치한다. 의존성 레이어 캐시를 살리기 위해 pom.xml 먼저 복사한다.
RUN apt-get update && apt-get install -y --no-install-recommends maven \
    && rm -rf /var/lib/apt/lists/*
COPY pom.xml .
RUN mvn -B -q dependency:go-offline

COPY tools ./tools
COPY src ./src
# captcha_data·db 는 이 빌드 컨텍스트 밖(repo root)이라 sync-resource 단계는 조용히 건너뛴다.
# 모델·DB 는 아래에서 호스트의 apps/springBoot/{models,db}/ 를 따로 COPY 해서 굽는다 —
# 미리 `mvn generate-resources` 등으로 채워 둬야 한다 (README 참고).
RUN mvn -B -q -DskipTests package && mv target/*.jar app.jar

FROM eclipse-temurin:25-jre AS runtime
WORKDIR /app

# HEALTHCHECK 용
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /build/app.jar ./app.jar
# 호스트의 apps/springBoot/{models,db}/ (mvn generate-resources 가 채워 둠) 를
# 이미지에 그대로 굽는다. 비어 있으면 빈 디렉터리만 복사된다 — 미리 채워 둘 것.
COPY models ./models
COPY db ./db

# non-root 로 띄운다. eclipse-temurin 베이스 이미지엔 이미 ubuntu(1000:1000) 가 있어서
# 그대로 쓴다 — 이 UID 가 개발 호스트의 일반 사용자 UID 와도 흔히 겹쳐 볼륨(logs)
# 권한 문제가 잘 안 생긴다. 호스트 UID 가 다르면 compose 에서 `user: "<uid>:<gid>"` 로 덮어쓴다.
# db/captchaSolver.sqlite3 는 기동 시 schema.sql 을 멱등 적용하느라 컨테이너 안에서 쓰기가
# 필요한데, 이 chown 이 그걸 ubuntu 소유로 만들어 준다 — 컨테이너의 쓰기 가능 레이어에만
# 남고 이미지 자체나 호스트 파일에는 반영되지 않는다(볼륨이 아니므로).
RUN chown -R ubuntu:ubuntu /app
USER ubuntu

# 모델·DB 모두 위에서 이미 이미지에 구웠다 — 볼륨 마운트 없음. 바꾸려면 apps/springBoot/
# {models,db}/ 를 다시 채우고(mvn generate-resources) 재빌드해야 한다.
# 컨테이너 안에서만 8080 을 쓴다(application.yml 기본값 5000 은 로컬 실행용, server.sh 와 맞춰 둔다).
ENV CAPTCHA_MODELS_DIR=/app/models \
    CAPTCHA_DB_PATH=/app/db/captchaSolver.sqlite3 \
    CAPTCHA_SCHEMA_PATH=/app/db/schema.sql \
    SERVER_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8080/ || exit 1

ENTRYPOINT ["java", "-jar", "app.jar"]
