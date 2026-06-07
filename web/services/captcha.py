import base64
import binascii
import os
import tempfile


_MODEL_CACHE = {}


class CaptchaPredictionError(Exception):
	pass


def get_model(captcha_id: str):
	if captcha_id in _MODEL_CACHE:
		return _MODEL_CACHE[captcha_id]

	import engine

	model = engine.get_captcha_model(captcha_id=captcha_id)
	_MODEL_CACHE[captcha_id] = model
	return model


def decode_image_data(image_data: str) -> bytes:
	if not isinstance(image_data, str) or not image_data.strip():
		raise ValueError("image_data is required")

	encoded = image_data.strip()
	if encoded.startswith("data:") and "," in encoded:
		encoded = encoded.split(",", 1)[1]

	try:
		return base64.b64decode(encoded, validate=True)
	except (binascii.Error, ValueError):
		raise ValueError("image_data must be valid base64")


def predict_from_bytes(captcha_id: str, image_bytes: bytes, filename: str = "captcha.png"):
	if not image_bytes:
		raise ValueError("image data is empty")

	import engine

	safe_filename = os.path.basename(filename) or "captcha.png"
	with tempfile.TemporaryDirectory() as td:
		tmp_path = os.path.join(td, safe_filename)
		try:
			with open(tmp_path, "wb") as out_f:
				out_f.write(image_bytes)
			model = get_model(captcha_id)
			return engine.predict(model=model, image_path=tmp_path, verbose=0)
		except Exception as e:
			raise CaptchaPredictionError(str(e)) from e
