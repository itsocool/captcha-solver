import os

cpu_only = os.getenv('CPU_ONLY', '1') == '1'

if cpu_only:
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
from flask import Flask, request, jsonify, render_template, send_file, abort
from PIL import Image
from werkzeug.utils import secure_filename
from captchaResolver import engine
from captchaResolver.dataclass import CaptchaType, TrainData
from captchaResolver.keras_core import KerasModel
import json
from flask import Response

captcha_id = 'gov24'
backend = 'keras'
rev = 1
image_width = 200
image_height = 50

captcha_type: CaptchaType = engine.get_captcha_type_list(backend=backend, cpu_only=cpu_only)[captcha_id]
model: KerasModel = KerasModel(captcha_type=captcha_type, verbose=1)
train_data: TrainData = model.train_data
model.train_data.rev = rev
model.train_data.image_width = image_width
model.train_data.image_height = image_height

app = Flask(__name__)

model.load_prediction_model(cpu_only=True)

@app.route("/")
@app.route("/captcha")
def index():
    # Render a small test page that lets users upload an image and see prediction
    return render_template('captcha.html', cpu_only=cpu_only)

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

def image_preprocess(image: Image.Image) -> Image.Image:
    if image.mode != 'RGB' and image.mode != 'L':
        image = image.convert('RGB')

    image = image.crop((1, 1, image.width, image.height))
    image = image.resize((image_width, image_height), Image.Resampling.LANCZOS)

    return image

@app.route("/api/v1/captcha", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "no file part 'image' in request"}), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify({"error": "no selected file"}), 400

    filename = secure_filename(f.filename)
    label_guess = os.path.splitext(filename)[0]

    stream = f.stream
    image: Image.Image = Image.open(stream)
 
    if image.size != (image_width, image_height):
        image = image_preprocess(image)
    # Measure processing time (ms)
    start = time.perf_counter()

    image_path = f"/tmp/{time.time()}.png"
    image.save(image_path)
    pred, confidence = engine.predict(model=model, image_path=image_path)

    elapsed_ms = int(round((time.perf_counter() - start) * 1000))

    image.close()
    os.remove(image_path)
    return jsonify({"predicted": pred, "confidence": confidence, "processing_ms": elapsed_ms})


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Captcha Resolver Web Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    args = parser.parse_args()
    
    # 전역 변수에 현재 포트 저장
    current_port = args.port
    
    # Check if we're in production environment
    is_production = os.getenv('FLASK_ENV') == 'production' or os.path.exists('/.dockerenv')
    
    # When run directly: start Flask development server
    app.run(host='0.0.0.0', port=args.port, debug=not is_production)

