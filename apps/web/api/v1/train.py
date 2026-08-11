import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from web.services.train import (
	PARAM_SPEC,
	TrainBusy,
	TrainError,
	clean_params,
	find_target,
	is_running,
	list_targets,
	run,
)


router = APIRouter(tags=["api-v1"])


@router.get("/train/targets")
async def train_targets():
	return JSONResponse({"targets": list_targets(), "running": is_running()})


def _sse(event: str, payload: dict) -> str:
	return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream(captcha_id: str, rev: int, device: str | None, params: dict):
	"""SSE 프레임 제너레이터.

	batch.py 와 같은 이유로 async 가 아닌 일반 def 다. 여기서는 학습 자체가
	워커 스레드에 있지만, 큐를 기다리는 get() 이 블로킹이라 마찬가지다.
	"""
	try:
		for event in run(captcha_id, rev, device, params):
			yield _sse(event["type"], event)
	except TrainBusy as e:
		# is_running() 확인과 실제 락 획득 사이에 다른 요청이 끼어든 경우.
		yield _sse("error", {"message": str(e)})
	except (ValueError, TrainError) as e:
		# 스트림이 이미 열린 뒤라 HTTP 상태 코드를 못 바꾼다. 오류도 이벤트로 보낸다.
		yield _sse("error", {"message": str(e)})
	except Exception as e:
		yield _sse("error", {"message": f"{type(e).__name__}: {e}"})


@router.get("/train/stream")
async def train_stream(
	request: Request,
	captcha_id: str = Query(...),
	rev: int = Query(0),
	device: str | None = Query(None),
):
	"""학습 진행 상황 SSE. EventSource 가 GET 만 지원해서 GET 이다.

	학습 파라미터는 이름이 여럿이라 하나씩 Query 로 받는 대신 쿼리스트링을
	통째로 넘긴다. 검증과 기본값 채우기는 services/train.clean_params 한 곳에서 한다.
	"""
	params_raw = {name: request.query_params.get(name) for name in PARAM_SPEC}
	params_raw["loss_type"] = request.query_params.get("loss_type")
	params_raw["use_amp"] = request.query_params.get("use_amp")
	params_raw["shuffle"] = request.query_params.get("shuffle")

	# 시작 전에 확인 가능한 오류는 스트림을 열기 전에 상태 코드로 알린다.
	try:
		from web.core.device import resolve as resolve_device

		target = find_target(captcha_id, rev)
		if not target["selectable"]:
			raise HTTPException(status_code=400, detail=target["reason"] or "학습할 수 없는 대상입니다")
		resolve_device(device)
		params = clean_params(params_raw)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	if is_running():
		raise HTTPException(status_code=409, detail="이미 다른 학습이 실행 중입니다")

	return StreamingResponse(
		_stream(captcha_id, rev, device, params),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			# nginx 등 리버스 프록시가 버퍼링하면 진행률이 끝나서야 한꺼번에 도착한다.
			"X-Accel-Buffering": "no",
		},
	)
