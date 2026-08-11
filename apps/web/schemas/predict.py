from pydantic import BaseModel, Field


class PredictJsonRequest(BaseModel):
	captcha_id: str | None = None
	image_data: str
	# 'auto' | 'cpu' | 'cuda'. 생략하면 auto (CUDA 가 있으면 CUDA, 없으면 CPU).
	device: str | None = Field(default=None, examples=["auto", "cpu", "cuda"])


class PredictResponse(BaseModel):
	captcha_id: str
	prediction: str
	confidence: float
	elapsed_ms: int
	# 실제 추론이 돌아간 디바이스. auto 로 요청했을 때 무엇이 선택됐는지 확인용.
	device: str
