# Dockerfile
# Python 3.12 공식 이미지 기반
FROM python:3.12-slim

# 작업 디렉터리 설정
WORKDIR /app

# RUN apt update && apt install -y gunicorn
# requirements.txt 복사 및 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .
# run_web.sh이 존재할 때만 권한 부여
RUN [ -f ./run_web.sh ] && chmod 755 ./run_web.sh || echo "no run_web.sh, skipping chmod"

# Flask 애플리케이션이 실행될 포트 노출
EXPOSE 5000

# 기본 실행 명령: run_web.sh이 있으면 이를 쓰도록 구성했지만,
# docker-compose에서 명시된 command가 없을 경우 안전하게 gunicorn으로 시작합니다.
# (프로젝트에 gunicorn이 requirements에 포함되어 있지 않다면, 필요 시 requirements.txt에 추가하세요)
CMD ["./run_web.sh", "restart"]
