import tempfile
import os
import captchaResolver.engine as engine
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 간단한 모델 캐시: captcha_id -> PyTorchModel
_MODEL_CACHE = {}

# 환경 변수 기본값
DEFAULT_CAPTCHA_ID = os.environ.get('DEFAULT_CAPTCHA_ID', 'supreme_court')
FLASK_HOST = os.environ.get('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.environ.get('FLASK_PORT', '5000'))
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'true').lower() in ('1', 'true', 'yes')

INDEX_HTML = """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Captcha Predictor</title></head>
  <body>
	<h2>Captcha Predictor</h2>
	<form action="/predict" method="post" enctype="multipart/form-data">
	  <label>Captcha ID: <input type="text" name="captcha_id" value="supreme_court"></label><br/>
	  <label>Image: <input type="file" name="image"></label><br/>
	  <button type="submit">Predict</button>
	</form>
	<div id="result"></div>
  </body>
</html>
"""


def get_model(captcha_id: str):
	if captcha_id in _MODEL_CACHE:
		return _MODEL_CACHE[captcha_id]
	model = engine.get_captcha_model(captcha_id=captcha_id)
	_MODEL_CACHE[captcha_id] = model
	return model


@app.route('/')
def index():
	return render_template_string(INDEX_HTML)


@app.route('/health')
def health():
	return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
	if 'image' not in request.files:
		return jsonify({'error': 'no image file provided'}), 400

	f = request.files['image']
	if f.filename == '':
		return jsonify({'error': 'empty filename'}), 400

	captcha_id = request.form.get('captcha_id', DEFAULT_CAPTCHA_ID)

	filename = secure_filename(f.filename)
	with tempfile.TemporaryDirectory() as td:
		tmp_path = os.path.join(td, filename)
		f.save(tmp_path)

		try:
			model = get_model(captcha_id)
			# engine.predict returns (pred_text, confidence)
			pred_text, confidence = engine.predict(model=model, image_path=tmp_path, verbose=0)
		except Exception as e:
			return jsonify({'error': str(e)}), 500

	return jsonify({'captcha_id': captcha_id, 'prediction': pred_text, 'confidence': float(confidence)})


if __name__ == '__main__':
	# 개발용 실행: 환경변수로 제어 가능합니다.
	app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

