import os
import time
from flask import Flask, request, jsonify, render_template
from PIL import Image
from werkzeug.utils import secure_filename
from captchaResolver import engine
from captchaResolver.dataclass import TrainData
from captchaResolver.keras_core import KerasModel

captcha_id = 'gov24'
backend = 'keras'
rev = 1
image_width = 200
image_height = 50
model: KerasModel = engine.get_captcha_model(captcha_id=captcha_id, backend=backend)
train_data: TrainData = model.train_data
model.train_data.rev = rev
model.train_data.image_width = image_width
model.train_data.image_height = image_height

app = Flask(__name__)
model.load_prediction_model()

@app.route("/")
def index():
    # Render a small test page that lets users upload an image and see prediction
    return render_template('index.html')

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

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

    image_path = f"/tmp/{time.time()}.png"
    image.save(image_path)
    pred, confidence = engine.predict(model=model, image_path=image_path)
    image.close()
    os.remove(image_path)
    return jsonify({"predicted": pred, "confidence": confidence})


if __name__ == '__main__':
    # When run directly: start Flask development server
    app.run(host='0.0.0.0', port=5000)

