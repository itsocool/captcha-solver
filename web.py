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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captcha Resolver Web Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    args = parser.parse_args()
    current_port = args.port
    is_production = os.getenv('FLASK_ENV') == 'production' or os.path.exists('/.dockerenv')
    app.run(host='0.0.0.0', port=args.port, debug=not is_production)
