import os

cpu_only = True

if cpu_only:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"] 

# from calendar import c
import time
import argparse
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

@app.route("/chat")
def chat():
    # Render a small test page that lets users upload an image and see prediction
    return render_template('chat.html', cpu_only=cpu_only)

@app.route("/health")
def health():
    # JSON을 ensure_ascii=False로 직렬화하고 UTF-8 컨텐츠 타입을 명시해 반환합니다.
    payload = {"status": "ok", "msg": "서비스가 정상적으로 작동하고 있습니다."}
    body = json.dumps(payload, ensure_ascii=False)
    return Response(body, content_type="application/json; charset=utf-8")

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
    
    # When run directly: start Flask development server
    app.run(host='0.0.0.0', port=args.port, debug=True)

