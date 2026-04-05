FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (이미지 처리를 위한 라이브러리)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 실행 스크립트에 실행 권한 부여
RUN chmod +x ./run_web.sh

# 앱 포트 노출 (Uvicorn/ASGI 앱과 호환)
EXPOSE 5000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:5000/health/ || exit 1

# 앱 환경 변수 설정 (환경 변수 접두사: WEB_ 로 변경됨)
ENV WEB_APP=web.py
ENV WEB_RUN_HOST=0.0.0.0

# 애플리케이션 실행
CMD ["./run_web.sh", "start"]
