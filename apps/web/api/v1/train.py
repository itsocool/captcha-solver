import json

import anyio
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from web.services.train import (
	PARAM_SPEC,
	TrainBusy,
	TrainError,
	current_session,
	is_running,
	list_targets,
	load_params,
	request_stop,
	save_params,
	start,
)


router = APIRouter(tags=["api-v1"])


@router.get("/train/targets")
async def train_targets():
	session = current_session()
	active = None
	if session is not None and not session.done.is_set():
		# 이어보기용: 어떤 대상이 돌고 있는지 알려 준다.
		active = {"captcha_id": session.captcha_id, "rev": session.rev}
	return JSONResponse({"targets": list_targets(), "running": is_running(), "active": active})


@router.get("/train/params")
async def train_params(captcha_id: str = Query(...), rev: int = Query(0)):
	"""대상(캡차, 리비전)의 저장된 학습 파라미터. 없으면 기본값을 돌려준다."""
	return JSONResponse({"params": load_params(captcha_id, rev)})


@router.post("/train/params")
async def train_params_save(request: Request, captcha_id: str = Query(...), rev: int = Query(0)):
	"""폼 파라미터를 저장한다. 학습을 시작하지 않아도 대상별로 유지된다.

	페이지를 떠났다 돌아와도 값이 남도록, 편집 시점에 프런트가 이걸 호출한다.
	"""
	raw = {name: request.query_params.get(name) for name in PARAM_SPEC}
	raw["loss_type"] = request.query_params.get("loss_type")
	raw["use_amp"] = request.query_params.get("use_amp")
	try:
		saved = save_params(captcha_id, rev, raw)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	return JSONResponse({"saved": True, "params": saved})


@router.post("/train/start")
async def train_start(
	request: Request,
	captcha_id: str = Query(...),
	rev: int = Query(0),
	device: str | None = Query(None),
):
	"""백그라운드 학습을 시작한다. 진행 상황은 GET /train/stream 으로 본다.

	학습은 SSE 연결과 분리된 워커 스레드에서 돈다 — 페이지를 벗어나도 계속되고,
	다시 열면 stream 이 히스토리를 재생한다. 파라미터는 이름이 여럿이라 쿼리스트링을
	통째로 넘기고, 검증·기본값 채우기는 services/train.clean_params 한 곳에서 한다.
	"""
	params_raw = {name: request.query_params.get(name) for name in PARAM_SPEC}
	params_raw["loss_type"] = request.query_params.get("loss_type")
	params_raw["use_amp"] = request.query_params.get("use_amp")
	params_raw["shuffle"] = request.query_params.get("shuffle")

	try:
		start(captcha_id, rev, device, params_raw)
	except TrainBusy as e:
		raise HTTPException(status_code=409, detail=str(e))
	except (ValueError, TrainError) as e:
		raise HTTPException(status_code=400, detail=str(e))

	return JSONResponse({"started": True, "captcha_id": captcha_id, "rev": rev})


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


async def _stream_session(request: Request):
	"""현재 세션에 붙어 히스토리를 재생하고 실시간 진행률을 흘린다.

	연결이 끊겨도 학습은 취소하지 않는다 (백그라운드에서 계속). 끊긴 연결에 대고
	쓰지 않도록 이벤트를 받은 뒤 is_disconnected 를 확인하고, 세션 stream 은
	버퍼에 이벤트를 남겨 두므로 다시 접속하면 처음부터 다시 재생된다.
	"""
	session = current_session()
	if session is None:
		yield _sse("idle", {"message": "진행 중인 학습이 없습니다"})
		return

	frames = session.stream()
	try:
		while True:
			# 세션 stream 은 이벤트/종료까지 블로킹하므로 스레드에서 한 개씩 받는다.
			event = await anyio.to_thread.run_sync(lambda: next(frames, None))
			if event is None:
				break
			if await request.is_disconnected():
				break
			yield _sse(event["type"], event)
	finally:
		frames.close()


@router.get("/train/stream")
async def train_stream(request: Request):
	"""진행 중(또는 방금 끝난) 학습 세션 SSE. EventSource 가 GET 만 지원해서 GET 이다.

	시작은 POST /train/start 가 한다. 이 엔드포인트는 붙어서 보기만 한다 — 파라미터가
	없고, 여러 탭이 동시에 붙어도 각자 히스토리를 재생받는다.
	"""
	return StreamingResponse(
		_stream_session(request),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			# nginx 등 리버스 프록시가 버퍼링하면 진행률이 끝나서야 한꺼번에 도착한다.
			"X-Accel-Buffering": "no",
		},
	)
