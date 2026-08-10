from time import perf_counter

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from web.core.db import get_service_config
from web.schemas.predict import PredictJsonRequest, PredictResponse
from web.services.captcha import (
	CaptchaPredictionError,
	decode_image_data,
	predict_from_bytes,
)


router = APIRouter(tags=["api-v1"])


def _response(captcha_id: str, prediction: str, confidence: float, started: float) -> PredictResponse:
	return PredictResponse(
		captcha_id=captcha_id,
		prediction=prediction,
		confidence=float(confidence),
		elapsed_ms=round((perf_counter() - started) * 1000),
	)


@router.post("/predictImage", response_model=PredictResponse)
async def predict_image(
	captcha_id: str | None = Form(None),
	image: UploadFile = File(...),
):
	started = perf_counter()

	if not image or not image.filename:
		raise HTTPException(status_code=400, detail="no image file provided")

	selected_captcha_id = captcha_id or get_service_config()["default_captcha_id"]
	image_bytes = await image.read()

	try:
		prediction, confidence = predict_from_bytes(selected_captcha_id, image_bytes, image.filename)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CaptchaPredictionError as e:
		raise HTTPException(status_code=500, detail=str(e))

	return _response(selected_captcha_id, prediction, confidence, started)


@router.post("/predictJson", response_model=PredictResponse)
async def predict_json(payload: PredictJsonRequest):
	started = perf_counter()
	selected_captcha_id = payload.captcha_id or get_service_config()["default_captcha_id"]

	try:
		image_bytes = decode_image_data(payload.image_data)
		prediction, confidence = predict_from_bytes(selected_captcha_id, image_bytes)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CaptchaPredictionError as e:
		raise HTTPException(status_code=500, detail=str(e))

	return _response(selected_captcha_id, prediction, confidence, started)
