import os

from torch import device

import ollama_generate
DEVICE = os.getenv('DEVICE', 'cpu')

if DEVICE == 'cpu':
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "none"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
else:
    if "NVIDIA_VISIBLE_DEVICES" in os.environ:
        del os.environ["NVIDIA_VISIBLE_DEVICES"] 
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"] 
    if "TF_CPP_MIN_LOG_LEVEL" in os.environ:
        del os.environ["TF_CPP_MIN_LOG_LEVEL"] 

import time
import argparse
import requests
import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
from captchaResolver import engine
from captchaResolver.dataclass import TrainData
from captchaResolver.core import PyTorchModel
from captchaResolver.keras_core import KerasModel
from flask import Response
from io import BytesIO

captcha_id = os.getenv('CAPTCHA_ID', 'gov24')
backend = os.getenv('BACKEND', 'pytorch')  # 'pytorch' or 'keras'
rev = int(os.getenv('REV', '1'))
image_width = int(os.getenv('IMAGE_WIDTH', '200'))
image_height = int(os.getenv('IMAGE_HEIGHT', '50'))

if backend == 'pytorch':
    model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
elif backend == 'keras':
    model: KerasModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
    
train_data: TrainData = model.train_data
train_data.rev = rev
train_data.image_width = image_width
train_data.image_height = image_height

app = Flask(__name__)

model.load_prediction_model()

@app.route("/")
@app.route("/captcha")
def index():
    # Render a small test page that lets users upload an image and see prediction
    return render_template(
        'captcha.html',
        device=DEVICE.upper(),
        backend=backend,
        captcha_id=captcha_id,
        rev=rev,
        image_width=image_width,
        image_height=image_height
    )

@app.route("/img")
def img():
    return render_template('img.html')

@app.route("/health/")
@app.route("/health/<int:port>")
@app.route("/health/<int:port>/")
def health(port=None):
    runtime_port = os.getenv('HOST_PORT', '5000')
    print(f"runtime_port: {runtime_port}, requested port: {port}")
    # port가 지정되지 않았거나 현재 포트와 같으면 자신의 상태 반환
    if port is None or port == runtime_port:
        payload = {"status": "ok", "msg": "서비스가 정상적으로 작동하고 있습니다.", "port": runtime_port}
        body = json.dumps(payload, ensure_ascii=False)
        response = Response(body, content_type="application/json; charset=utf-8")
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    # 다른 포트가 요청되면 해당 포트로 프록시 요청
    try:
        target_url = f"http://192.168.50.98:{port}/health/"
        proxy_response = requests.get(target_url, timeout=3)
        
        response = Response(
            proxy_response.content,
            status=proxy_response.status_code,
            content_type="application/json; charset=utf-8"
        )
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except requests.exceptions.RequestException as e:
        # 연결 실패 시 오류 상태 반환
        payload = {"status": "error", "msg": f"포트 {port} 서버에 연결할 수 없습니다.", "error": str(e)}
        body = json.dumps(payload, ensure_ascii=False)
        response = Response(body, status=503, content_type="application/json; charset=utf-8")
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

def predict(image: Image.Image):
    if image.size != (image_width, image_height):
        if image.mode != 'RGB' and image.mode != 'L':
            image = image.convert('RGB')

        image = image.crop((1, 1, image.width, image.height))
        image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)

    start = time.perf_counter()
    image_path = f"/tmp/{time.time()}.png"
    image.save(image_path)
    pred, confidence = engine.predict(model=model, image_path=image_path)
    elapsed_ms = int(round((time.perf_counter() - start) * 1000))
    image.close()
    os.remove(image_path)
    return pred, confidence, elapsed_ms

def ocr(image: Image.Image):
    if image.size != (image_width, image_height):
        if image.mode != 'RGB' and image.mode != 'L':
            image = image.convert('RGB')

        image = image.crop((1, 1, image.width, image.height))
        image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)

    image_path = f"/tmp/{time.time()}.png"
    image.save(image_path)
    result = ollama_generate.ocr_image(image_path=image_path)
    pred, confidence, elapsed_ms, bbox = result.get("text"), result.get("confidence"), result.get("processing_ms"), result.get("bbox")
    image.close()
    os.remove(image_path)
    return pred, confidence, elapsed_ms, bbox

# @app.route("/api/v1/chat", methods=["POST"])
# def chat():
#     message = request.json.get("message", "")
#     result = ollama_generate.chat_with_image(message=message)
#     response = result.get("response", "")
#     confidence = result.get("confidence", 0.0)
#     elapsed_ms = result.get("processing_ms", 0)

#     return response, confidence, elapsed_ms

@app.route("/api/v1/captcha", methods=["POST"])
def predict_multi_part():
    if 'image' not in request.files:
        return jsonify({"error": "no file part 'image' in request"}), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify({"error": "no selected file"}), 400

    try:
        stream = f.stream
        image: Image.Image = Image.open(stream)
        pred, confidence, elapsed_ms = predict(image)
        bbox = None
        return jsonify({"predicted": pred, "confidence": confidence, "processing_ms": elapsed_ms, "bbox": bbox})
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

@app.route("/api/v1/captcha/octet-stream", methods=["POST"])
def predict_octet_stream():
    if request.content_type != 'application/octet-stream':
        return jsonify({"error": "Content-Type must be application/octet-stream"}), 400
    
    binary_data = request.get_data()
    
    if not binary_data:
        return jsonify({"error": "no binary data received"}), 400
    
    try:
        image_stream = BytesIO(binary_data)
        image: Image.Image = Image.open(image_stream)
        pred, confidence, elapsed_ms = predict(image)
        return jsonify({"predicted": pred, "confidence": confidence, "processing_ms": elapsed_ms})     
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

@app.route("/api/v1/ocr", methods=["POST"])
def ocr_multi_part():
    try:
        if 'image' not in request.files:
            error_msg = "no file part 'image' in request"
            print(f"OCR Error: {error_msg}")
            return jsonify({"error": error_msg}), 400

        f = request.files['image']
        if f.filename == '':
            error_msg = "no selected file"
            print(f"OCR Error: {error_msg}")
            return jsonify({"error": error_msg}), 400

        try:
            stream = f.stream
            image: Image.Image = Image.open(stream)
            pred, confidence, elapsed_ms, bbox = ocr(image)
            return jsonify({"predicted": pred, "confidence": confidence, "processing_ms": elapsed_ms, "bbox": bbox})
        except Exception as e:
            import traceback
            error_msg = f"Failed to process image: {str(e)}"
            print(f"OCR Processing Error: {error_msg}")
            print(traceback.format_exc())
            return jsonify({"error": error_msg, "trace": traceback.format_exc()}), 500
    except Exception as e:
        import traceback
        error_msg = f"Unexpected error: {str(e)}"
        print(f"OCR Unexpected Error: {error_msg}")
        print(traceback.format_exc())
        return jsonify({"error": error_msg, "trace": traceback.format_exc()}), 500

def _load_context_library(library: str | None) -> str | None:
    """Load cached Context7 document for the specified library."""
    if not library:
        return None
    
    try:
        cached_dir = os.path.join(os.path.dirname(__file__), 'cached_contexts')
        cached_path = os.path.join(cached_dir, f"{library}.txt")
        if os.path.exists(cached_path):
            with open(cached_path, 'r', encoding='utf-8') as fh:
                return fh.read()
    except Exception:
        pass
    
    return None


def _prepare_full_message(message: str, library: str | None) -> str:
    """Prepare the full message with optional context prepended."""
    context_text = _load_context_library(library)
    if context_text:
        return f"[Context from {library}]\n{context_text}\n\nUser message:\n{message}"
    return message


def _handle_streaming_response(full_message: str) -> Response:
    """Handle streaming chat response - expects plain text chunks only."""
    gen = ollama_generate.chat_text(message=full_message, stream=True)

    def stream_generator():
        try:
            for part in gen:
                # Streaming mode should return plain text only
                # No JSON parsing needed as the model is instructed to return plain text
                text_chunk = ''
                
                try:
                    if isinstance(part, str):
                        text_chunk = part
                    elif isinstance(part, dict):
                        # Fallback: if dict received, extract text field
                        text_chunk = part.get('text') or part.get('content') or part.get('response') or str(part)
                    else:
                        text_chunk = str(part)
                except Exception:
                    continue
                
                text_chunk = text_chunk.strip()
                if text_chunk:
                    yield text_chunk
                    
        except Exception as e:
            try:
                yield f"[error] {str(e)}"
            except Exception:
                yield "[error]"

    # Return a chunked text/plain response (no JSON) for streaming clients
    response = Response(stream_generator(), content_type='text/plain; charset=utf-8')
    # disable caching for interactive stream
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


def _handle_normal_response(full_message: str):
    """Handle normal (non-streaming) chat response."""
    result = ollama_generate.chat_text(message=full_message)

    text = ""
    confidence = 0.0
    elapsed_ms = 0

    try:
        if isinstance(result, dict):
            # prefer common keys but allow fallbacks
            text = result.get("text") or result.get("response") or result.get("output") or ""
            try:
                confidence = float(result.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            elapsed_ms = int(result.get("processing_ms", result.get("processing_time_ms", result.get("processing_ms", 0))))
        elif isinstance(result, str):
            text = result
        elif isinstance(result, (list, tuple)):
            # try to find first string as response and numeric values as confidence/elapsed
            for item in result:
                if isinstance(item, str):
                    text = item
                    break
            nums = [i for i in result if isinstance(i, (int, float))]
            if len(nums) == 1:
                # ambiguous: treat single numeric as elapsed_ms
                elapsed_ms = int(nums[0])
            elif len(nums) >= 2:
                # assume first numeric is confidence, last is elapsed_ms
                try:
                    confidence = float(nums[0])
                except Exception:
                    confidence = 0.0
                try:
                    elapsed_ms = int(nums[-1])
                except Exception:
                    elapsed_ms = 0
        else:
            # unknown type: stringify
            text = str(result)
    except Exception:
        # if anything unexpected happens when parsing, fallback to safe defaults
        try:
            text = str(result)
        except Exception:
            text = ""
        confidence = 0.0
        elapsed_ms = 0

    # Return using the new chat response schema: { text, confidence, processing_ms }
    return jsonify({"text": text, "confidence": confidence, "processing_ms": elapsed_ms})


@app.route("/api/v1/chat", methods=["GET", "POST"])
def chat_text():
    """Simple text chat endpoint with optional image attachment.

    Usage:
      GET /api/v1/chat?message=hello
      GET /api/v1/chat?message=how+to+use+tf&library=tensorflow
      GET /api/v1/chat?message=hello&stream=1
      POST /api/v1/chat (with form-data: message=hello, image=<file>, stream=1)

    Query/Form Parameters:
      - message: The chat message (required if no image)
      - library: Optional library name for Context7 cached docs
      - stream: '1', 'true', or 'on' to enable streaming response
      - image: Optional image file (POST only)

    If `library` query param is provided, this endpoint will look for a cached
    Context7 document at `cached_contexts/<library>.txt` (relative to project root)
    and prepend it to the user's message before calling the chat helper.
    
    If an image is provided, it will be processed first using the captcha model,
    and the prediction will be included in the message context.
    """
    # Read params from both query and form
    message = request.args.get('message') or request.form.get('message') or ''
    library = request.args.get('library') or request.form.get('library') or None
    stream_flag = request.args.get('stream') or request.form.get('stream') or '0'
    
    # Handle image attachment if present
    image_context = ""
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename:
            try:
                # Process image with captcha model
                image = Image.open(file.stream)
                start_time = time.time()
                predicted, confidence = model.predict(image)
                processing_ms = int((time.time() - start_time) * 1000)
                
                image_context = f"\n\n[이미지 분석 결과]\n예측값: {predicted}\n신뢰도: {confidence:.4f}\n처리시간: {processing_ms}ms\n\n"
            except Exception as e:
                image_context = f"\n\n[이미지 처리 오류: {str(e)}]\n\n"
    
    # Prepare full message with image context and optional library context
    full_message = image_context + message
    full_message = _prepare_full_message(full_message, library)
    
    # Determine if client requested streaming
    use_stream = str(stream_flag).lower() in ('1', 'true', 'on')
    
    # Route to appropriate handler
    if use_stream:
        return _handle_streaming_response(full_message)
    else:
        return _handle_normal_response(full_message)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captcha Resolver Web Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    args = parser.parse_args()
    current_port = args.port
    is_production = os.getenv('FLASK_ENV') == 'production' or os.path.exists('/.dockerenv')
    app.run(host='0.0.0.0', port=args.port, debug=not is_production)
