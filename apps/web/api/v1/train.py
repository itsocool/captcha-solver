import json
import threading

import anyio
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
	load_params,
	request_stop,
	run,
)


router = APIRouter(tags=["api-v1"])


@router.get("/train/targets")
async def train_targets():
	return JSONResponse({"targets": list_targets(), "running": is_running()})


@router.get("/train/params")
async def train_params(captcha_id: str = Query(...), rev: int = Query(0)):
	"""대상(캡차, 리비전)의 저장된 학습 파라미터. 없으면 기본값을 돌려준다."""
	return JSONResponse({"params": load_params(captcha_id, rev)})


@router.post("/train/stop")
async def train_stop(save: bool = Query(True)):
	"""실행 중인 학습 중단 요청. save=false 면 best 모델을 저장하지 않고 버린다.

	에폭 경계에서 걸리므로 진행 중인 에폭 하나는 마저 돈다. 결과는 열려 있는
	SSE 스트림의 done/skipped 이벤트로 온다.
	"""
	if not request_stop(save=save):
		raise HTTPException(status_code=409, detail="실행 중인 학습이 없습니다")
	return JSONResponse({"stopping": True, "save": save})


def _sse(event: str, payload: dict) -> str:
	return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream(request: Request, captcha_id: str, rev: int, device: str | None, params: dict):
	"""SSE 프레임 제너레이터.

	batch.py 와 달리 async 다. 학습 제너레이터는 큐 대기로 블로킹되는데, 그러면
	Starlette 이 클라이언트 끊김을 알려줄 방법이 없다 (동기 호출로 막힌 스레드풀
	워커에는 close 가 도달하지 못한다). 그래서 한 이벤트씩 스레드에서 받아오고,
	매 이벤트 사이에 request.is_disconnected() 로 직접 확인해 취소를 건다.

	확인 주기는 이벤트 하나(= 에폭 하나)다. 학습 중단은 원래 에폭 경계에서만
	가능하므로 더 촘촘히 볼 이유가 없다.
	"""
	cancel = threading.Event()
	events = run(captcha_id, rev, device, params, cancel=cancel)

	try:
		while True:
			if await request.is_disconnected():
				cancel.set()
				break
			try:
				event = await anyio.to_thread.run_sync(lambda: next(events, None))
			except StopIteration:
				break
			if event is None:
				break
			yield _sse(event["type"], event)
	except TrainBusy as e:
		# is_running() 확인과 실제 락 획득 사이에 다른 요청이 끼어든 경우.
		yield _sse("error", {"message": str(e)})
	except (ValueError, TrainError) as e:
		# 스트림이 이미 열린 뒤라 HTTP 상태 코드를 못 바꾼다. 오류도 이벤트로 보낸다.
		yield _sse("error", {"message": str(e)})
	except Exception as e:
		yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
	finally:
		cancel.set()
		events.close()


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
		_stream(request, captcha_id, rev, device, params),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			# nginx 등 리버스 프록시가 버퍼링하면 진행률이 끝나서야 한꺼번에 도착한다.
			"X-Accel-Buffering": "no",
		},
	)
