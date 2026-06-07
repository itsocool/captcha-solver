from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from web.core.config import get_settings
from web.schemas.predict import PredictJsonRequest, PredictResponse
from web.services.captcha import (
	CaptchaPredictionError,
	decode_image_data,
	predict_from_bytes,
)


router = APIRouter(tags=["api-v1"])


def _response(captcha_id: str, prediction: str, confidence: float) -> PredictResponse:
	return PredictResponse(
		captcha_id=captcha_id,
		prediction=prediction,
		confidence=float(confidence),
	)


@router.post("/predictImage", response_model=PredictResponse)
async def predict_image(
	captcha_id: str | None = Form(None),
	image: UploadFile = File(...),
):
	if not image or not image.filename:
		raise HTTPException(status_code=400, detail="no image file provided")

	settings = get_settings()
	selected_captcha_id = captcha_id or settings.default_captcha_id
	image_bytes = await image.read()

	try:
		prediction, confidence = predict_from_bytes(selected_captcha_id, image_bytes, image.filename)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CaptchaPredictionError as e:
		raise HTTPException(status_code=500, detail=str(e))

	return _response(selected_captcha_id, prediction, confidence)


@router.post("/predictJson", response_model=PredictResponse)
async def predict_json(payload: PredictJsonRequest):
	settings = get_settings()
	selected_captcha_id = payload.captcha_id or settings.default_captcha_id

	try:
		image_bytes = decode_image_data(payload.image_data)
		prediction, confidence = predict_from_bytes(selected_captcha_id, image_bytes)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	except CaptchaPredictionError as e:
		raise HTTPException(status_code=500, detail=str(e))

	return _response(selected_captcha_id, prediction, confidence)
