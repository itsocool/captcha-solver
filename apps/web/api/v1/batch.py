import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from web.services.batch_predict import (
	BatchPredictBusy,
	BatchPredictError,
	is_running,
	list_targets,
	pred_image_path,
	run,
)


router = APIRouter(tags=["api-v1"])


@router.get("/batch/targets")
async def batch_targets():
	return JSONResponse({"targets": list_targets(), "running": is_running()})


def _sse(event: str, payload: dict) -> str:
	return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream(captcha_id: str, rev: int, device: str | None):
	"""SSE 프레임 제너레이터.

	async 가 아니라 일반 def 다. 추론은 CPU 를 붙잡는 동기 작업이라 async 제너레이터로
	두면 이벤트 루프가 통째로 막혀 서버 전체가 멈춘다. 동기 제너레이터를 주면
	Starlette 가 스레드풀에서 돌려준다.
	"""
	try:
		for event in run(captcha_id, rev, device):
			yield _sse(event["type"], event)
	except BatchPredictBusy as e:
		# is_running() 확인과 실제 락 획득 사이에 다른 요청이 끼어든 경우.
		yield _sse("error", {"message": str(e)})
	except (ValueError, BatchPredictError) as e:
		# 스트림이 이미 열린 뒤라 HTTP 상태 코드를 못 바꾼다. 오류도 이벤트로 보낸다.
		yield _sse("error", {"message": str(e)})
	except Exception as e:
		yield _sse("error", {"message": f"{type(e).__name__}: {e}"})


@router.get("/batch/stream")
async def batch_stream(
	captcha_id: str = Query(...),
	rev: int = Query(1),
	device: str | None = Query(None),
):
	"""일괄 추론 진행 상황 SSE. EventSource 가 GET 만 지원해서 GET 이다."""
	# 시작 전에 확인 가능한 오류는 스트림을 열기 전에 상태 코드로 알린다.
	try:
		from web.services.batch_predict import find_target
		from web.core.device import resolve as resolve_device

		target = find_target(captcha_id, rev)
		if not target["selectable"]:
			raise HTTPException(status_code=400, detail=target["reason"] or "실행할 수 없는 대상입니다")
		resolve_device(device)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	if is_running():
		raise HTTPException(status_code=409, detail="이미 다른 일괄 추론이 실행 중입니다")

	return StreamingResponse(
		_stream(captcha_id, rev, device),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			# nginx 등 리버스 프록시가 버퍼링하면 진행률이 끝나서야 한꺼번에 도착한다.
			"X-Accel-Buffering": "no",
		},
	)


@router.get("/batch/image")
async def batch_image(
	captcha_id: str = Query(...),
	rev: int = Query(1),
	name: str = Query(...),
):
	"""오답 갤러리용 pred 이미지 썸네일."""
	try:
		path = pred_image_path(captcha_id, rev, name)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	return FileResponse(path, media_type="image/png")
