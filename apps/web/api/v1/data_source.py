import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from web.services.data_source import (
	DataSourceBusy,
	DataSourceError,
	clean_request,
	draft_image_path,
	is_running,
	list_targets,
	run,
)


router = APIRouter(tags=["api-v1"])


@router.get("/data-source/targets")
async def data_source_targets():
	return JSONResponse({"targets": list_targets(), "running": is_running()})


def _sse(event: str, payload: dict) -> str:
	return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream(captcha_id: str, rev: int, url: str, selector: str, count: int):
	"""SSE 프레임 제너레이터.

	batch.py 와 같은 이유로 async 가 아닌 일반 def 다. 수집은 네트워크 대기가
	대부분인 동기 작업이라 async 제너레이터로 두면 이벤트 루프가 막힌다.
	동기 제너레이터를 주면 Starlette 이 스레드풀에서 돌려준다.
	"""
	try:
		for event in run(captcha_id, rev, url, selector, count):
			yield _sse(event["type"], event)
	except DataSourceBusy as e:
		# is_running() 확인과 실제 락 획득 사이에 다른 요청이 끼어든 경우.
		yield _sse("error", {"message": str(e)})
	except (ValueError, DataSourceError) as e:
		# 스트림이 이미 열린 뒤라 HTTP 상태 코드를 못 바꾼다. 오류도 이벤트로 보낸다.
		yield _sse("error", {"message": str(e)})
	except Exception as e:
		yield _sse("error", {"message": f"{type(e).__name__}: {e}"})


@router.get("/data-source/stream")
async def data_source_stream(
	captcha_id: str = Query(...),
	rev: int = Query(0),
	url: str = Query(...),
	selector: str = Query(""),
	count: int = Query(...),
):
	"""수집 진행 상황 SSE. EventSource 가 GET 만 지원해서 GET 이다."""
	# 시작 전에 확인 가능한 오류는 스트림을 열기 전에 상태 코드로 알린다.
	try:
		clean_request(captcha_id, rev, url, selector, count)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	if is_running():
		raise HTTPException(status_code=409, detail="이미 다른 수집이 실행 중입니다")

	return StreamingResponse(
		_stream(captcha_id, rev, url, selector, count),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			# nginx 등 리버스 프록시가 버퍼링하면 진행률이 끝나서야 한꺼번에 도착한다.
			"X-Accel-Buffering": "no",
		},
	)


@router.get("/data-source/image")
async def data_source_image(
	captcha_id: str = Query(...),
	rev: int = Query(0),
	name: str = Query(...),
):
	"""수집한 draft 이미지 썸네일."""
	try:
		path = draft_image_path(captcha_id, rev, name)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	return FileResponse(path, media_type="image/png")
