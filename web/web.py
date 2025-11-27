import os
import time
import argparse
import requests
import json
from flask import Flask, g, request, jsonify, render_template, send_file, abort
from PIL import Image
from torch import threshold
from werkzeug.utils import secure_filename
from captchaResolver import engine
from captchaResolver.core import PyTorchModel
from captchaResolver.dataclass import TrainData
from flask import Response

captcha_id = 'gov24'
backend = 'pytorch'
cpu_only = True
rev = 1
model: PyTorchModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
model.train_data.rev = rev

app = Flask(__name__)

model.load_prediction_model(cpu_only=cpu_only)

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

def image_preprocess(image: Image.Image, threshold: int = 255) -> Image.Image:
    image = image if image.mode == 'RGBA' else image.convert('RGBA')
    white_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    combined_image = Image.alpha_composite(white_background, image)
    grayscale_image = combined_image.convert("RGB").convert("L")
    processed_image = grayscale_image.point(lambda x: x if x < threshold else 255) if threshold < 255 else grayscale_image
    return processed_image

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

@app.route("/api/v1/captcha/octet-stream", methods=["POST"])
def predict_octet_stream():
    """
    Handles octet-stream binary file upload for captcha prediction
    Content-Type: application/octet-stream
    """
    # Check content type
    if request.content_type != 'application/octet-stream':
        return jsonify({"error": "Content-Type must be application/octet-stream"}), 400
    
    # Get raw binary data
    binary_data = request.get_data()
    
    if not binary_data:
        return jsonify({"error": "no binary data received"}), 400
    
    try:
        # Create image from binary data
        from io import BytesIO
        image_stream = BytesIO(binary_data)
        image: Image.Image = Image.open(image_stream)
        
        # Preprocess image if needed
        if image.size != (image_width, image_height):
            image = image_preprocess(image)
        
        # Measure processing time (ms)
        start = time.perf_counter()

        # Save temporary image for prediction
        image_path = f"/tmp/{time.time()}.png"
        image.save(image_path)
        pred, confidence = engine.predict(model=model, image_path=image_path)

        elapsed_ms = int(round((time.perf_counter() - start) * 1000))

        # Cleanup
        image.close()
        image_stream.close()
        os.remove(image_path)
        
        return jsonify({"predicted": pred, "confidence": confidence, "processing_ms": elapsed_ms})
        
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

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

