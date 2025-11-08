# Ollama OCR 설정 가이드

## 개요

이 프로젝트는 Ollama를 사용한 OCR(광학 문자 인식) 기능을 제공합니다. Ollama는 로컬 서버 또는 클라우드 API를 통해 사용할 수 있습니다.

## 설정 방법

### 1. 로컬 Ollama 서버 사용 (권장)

#### 1.1 Ollama 설치

```bash
# Linux/WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows
# https://ollama.com/download 에서 다운로드
```

#### 1.2 Ollama 서버 시작

```bash
ollama serve
```

기본적으로 `http://localhost:11434`에서 실행됩니다.

#### 1.3 비전 모델 다운로드

```bash
# LLaMA 3.2 Vision 모델 (권장)
ollama pull llama3.2-vision

# 또는 다른 비전 모델
ollama pull llava
ollama pull bakllava
```

#### 1.4 환경 변수 설정 (선택사항)

```bash
# 기본값이므로 설정하지 않아도 됩니다
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL_ID=llama3.2-vision
```

### 2. 클라우드 Ollama API 사용

> ⚠️ **주의**: `https://ollama.com/api`는 웹사이트이지 API 엔드포인트가 아닙니다.
> Ollama는 공식 클라우드 서비스를 제공하지 않습니다.

만약 커스텀 Ollama 서버를 사용한다면:

```bash
export OLLAMA_HOST=https://your-custom-ollama-server.com
export OLLAMA_MODEL_ID=your-model-name
export OLLAMA_API_KEY=your-api-key  # 필요한 경우
```

## 사용 방법

### Flask 웹 서버에서 사용

```bash
# .venv 환경 활성화
source .venv/bin/activate

# 서버 시작 (로컬 Ollama 사용)
python web.py

# 또는 커스텀 호스트 지정
OLLAMA_HOST=http://localhost:11434 python web.py
```

웹 UI에서 `/captcha` 페이지를 열고 "OCR" 버튼을 클릭하여 이미지를 업로드합니다.

### CLI에서 직접 테스트

```bash
# 로컬 Ollama 서버 사용
python ollama_generate.py --image /path/to/image.png

# 커스텀 호스트 및 모델 지정
python ollama_generate.py \
  --image /path/to/image.png \
  --host http://localhost:11434 \
  --model llama3.2-vision
```

### API 엔드포인트 호출

```bash
# OCR API 호출
curl -X POST http://localhost:5000/api/v1/ocr \
  -F "image=@/path/to/image.png"
```

응답 예시:
```json
{
  "predicted": "ABCD1234",
  "confidence": 0.95,
  "processing_ms": 1250
}
```

## 트러블슈팅

### 404 에러: `/public/` 경로

이 에러는 잘못된 Ollama 호스트 설정으로 인해 HTML 페이지가 반환될 때 발생합니다.

**해결 방법:**
1. `OLLAMA_HOST`가 올바른 API 엔드포인트인지 확인
2. 로컬 Ollama 서버가 실행 중인지 확인: `curl http://localhost:11434`
3. 모델이 다운로드되어 있는지 확인: `ollama list`

### Connection Refused 에러

```bash
# Ollama 서버 상태 확인
systemctl status ollama  # Linux systemd
ps aux | grep ollama     # 프로세스 확인

# Ollama 서버 시작
ollama serve
```

### Model Not Found 에러

```bash
# 사용 가능한 모델 확인
ollama list

# 모델 다운로드
ollama pull llama3.2-vision
```

### GPU 메모리 부족

```bash
# 더 작은 모델 사용
export OLLAMA_MODEL_ID=llava:7b

# 또는 CPU 전용 모드로 실행
OLLAMA_NUM_GPU=0 ollama serve
```

## 모델 비교

| 모델 | 크기 | 속도 | 정확도 | 권장 용도 |
|------|------|------|--------|-----------|
| llama3.2-vision | ~7GB | 중간 | 높음 | 일반 OCR |
| llava | ~4GB | 빠름 | 중간 | 빠른 처리 필요 시 |
| bakllava | ~8GB | 느림 | 매우 높음 | 고정확도 필요 시 |

## 성능 최적화

1. **GPU 사용**: CUDA를 지원하는 GPU가 있다면 자동으로 사용됩니다
2. **모델 프리로딩**: 서버 시작 시 `keep_alive` 옵션으로 모델을 메모리에 유지
3. **배치 처리**: 여러 이미지를 순차적으로 처리할 때 모델이 언로드되지 않도록 설정

```python
# web.py 또는 ollama_generate.py에서
response = client.chat(
    model=model_id,
    messages=messages,
    options={'temperature': 0.0},
    keep_alive='5m'  # 5분 동안 모델을 메모리에 유지
)
```

## 추가 참고 자료

- Ollama 공식 문서: https://github.com/ollama/ollama
- Ollama Python SDK: https://github.com/ollama/ollama-python
- 지원 모델 목록: https://ollama.com/library
