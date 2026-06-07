from pydantic import BaseModel


class PredictJsonRequest(BaseModel):
	captcha_id: str | None = None
	image_data: str


class PredictResponse(BaseModel):
	captcha_id: str
	prediction: str
	confidence: float
