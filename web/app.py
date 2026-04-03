import tempfile
import os
import captchaResolver.engine as engine
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Captcha Predictor")

# 간단한 모델 캐시: captcha_id -> PyTorchModel
_MODEL_CACHE = {}

# 환경 변수 기본값 (환경 변수 접두사를 WEB_로 변경했습니다)
DEFAULT_CAPTCHA_ID = os.environ.get('DEFAULT_CAPTCHA_ID', 'supreme_court')
APP_HOST = os.environ.get('WEB_HOST', '0.0.0.0')
APP_PORT = int(os.environ.get('WEB_PORT', '5000'))
APP_DEBUG = os.environ.get('WEB_DEBUG', 'true').lower() in ('1', 'true', 'yes')

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


@app.get('/', response_class=HTMLResponse)
async def index():
	return HTMLResponse(content=INDEX_HTML)


@app.get('/health')
@app.get('/health/')
async def health():
	return JSONResponse({'status': 'ok'})


@app.post('/predict')
async def predict(captcha_id: str = Form(DEFAULT_CAPTCHA_ID), image: UploadFile = File(...)):
	if not image or not image.filename:
		raise HTTPException(status_code=400, detail='no image file provided')

	filename = os.path.basename(image.filename)
	with tempfile.TemporaryDirectory() as td:
		tmp_path = os.path.join(td, filename)
		contents = await image.read()
		try:
			with open(tmp_path, 'wb') as out_f:
				out_f.write(contents)
			model = get_model(captcha_id)
			# engine.predict returns (pred_text, confidence)
			pred_text, confidence = engine.predict(model=model, image_path=tmp_path, verbose=0)
		except Exception as e:
			raise HTTPException(status_code=500, detail=str(e))

	return JSONResponse({'captcha_id': captcha_id, 'prediction': pred_text, 'confidence': float(confidence)})


if __name__ == '__main__':
	# 개발용 실행: uvicorn으로 실행됩니다.
	import uvicorn
	uvicorn.run(app, host=APP_HOST, port=APP_PORT, reload=APP_DEBUG)

