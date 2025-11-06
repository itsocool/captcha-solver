FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (TensorFlow 및 이미지 처리를 위한 라이브러리)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Flask 포트 노출
EXPOSE 5000

# Flask 환경 변수 설정
ENV FLASK_APP=web.py
ENV FLASK_RUN_HOST=0.0.0.0

# 애플리케이션 실행
CMD ["python", "web.py"]
