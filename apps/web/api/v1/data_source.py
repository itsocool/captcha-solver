import json
from contextlib import closing

import anyio

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from web.services.data_source import (
	DataSourceBusy,
	DataSourceError,
	clean_request,
	draft_image_path,
	is_running,
	iter_auto_label,
	list_drafts,
	list_targets,
	load_params,
	move_checked_to_train,
	rename_draft,
	run,
	save_params,
)


router = APIRouter(tags=["api-v1"])


class TaskStreamingResponse(StreamingResponse):
	"""연결 종료 시 동기 작업 제너레이터를 닫아 작업 락을 즉시 반환한다."""

	def __init__(self, content, **kwargs):
		self.worker = content
		super().__init__(content, **kwargs)

	async def __call__(self, scope, receive, send):
		try:
			await super().__call__(scope, receive, send)
		finally:
			# 취소된 응답도 정리를 완료해야 한다. Starlette 스레드풀의 진행 중
			# next()가 끝난 뒤 닫으므로 실행 중인 제너레이터를 강제 해제하지 않는다.
			with anyio.CancelScope(shield=True):
				await run_in_threadpool(self.worker.close)


@router.get("/data-source/targets")
async def data_source_targets():
	return JSONResponse({"targets": list_targets(), "running": is_running()})


@router.get("/data-source/params")
async def data_source_params(captcha_id: str = Query(...), rev: int = Query(1)):
	"""대상(캡차, 리비전)의 저장된 수집 입력값. 없으면 기본값을 돌려준다."""
	return JSONResponse({"params": load_params(captcha_id, rev)})


@router.post("/data-source/params")
async def data_source_params_save(request: Request, captcha_id: str = Query(...), rev: int = Query(1)):
	"""폼 입력값을 저장한다. 수집을 시작하지 않아도 대상별로 유지된다 (편집 시 프런트가 호출)."""
	raw = {name: request.query_params.get(name) for name in ("url", "content_type", "selector", "count", "delay_ms")}
	try:
		saved = save_params(captcha_id, rev, raw)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))
	return JSONResponse({"saved": True, "params": saved})


def _sse(event: str, payload: dict) -> str:
	return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream(captcha_id: str, rev: int, url: str, selector: str, count: int, delay_ms: int, content_type: str):
	"""SSE 프레임 제너레이터.

	batch.py 와 같은 이유로 async 가 아닌 일반 def 다. 수집은 네트워크 대기가
	대부분인 동기 작업이라 async 제너레이터로 두면 이벤트 루프가 막힌다.
	동기 제너레이터를 주면 Starlette 이 스레드풀에서 돌려준다.
	"""
	try:
		with closing(run(captcha_id, rev, url, selector, count, delay_ms, content_type)) as events:
			for event in events:
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
	rev: int = Query(1),
	url: str = Query(...),
	selector: str = Query(""),
	count: int = Query(...),
	delay_ms: int = Query(0),
	content_type: str = Query("image"),
):
	"""수집 진행 상황 SSE. 기존 GET 요청 URL을 유지한다."""
	# 시작 전에 확인 가능한 오류는 스트림을 열기 전에 상태 코드로 알린다.
	try:
		clean_request(captcha_id, rev, url, selector, count, delay_ms, content_type)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	if is_running():
		raise HTTPException(status_code=409, detail="이미 다른 수집이 실행 중입니다")

	return TaskStreamingResponse(
		_stream(captcha_id, rev, url, selector, count, delay_ms, content_type),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			# nginx 등 리버스 프록시가 버퍼링하면 진행률이 끝나서야 한꺼번에 도착한다.
			"X-Accel-Buffering": "no",
		},
	)


def _auto_label_stream(captcha_id: str, rev: int, device: str | None, min_confidence: float, rename_files: bool = True):
	try:
		with closing(iter_auto_label(captcha_id, rev, device, min_confidence, rename_files=rename_files)) as events:
			for event in events:
				yield _sse(event["type"], event)
	except DataSourceBusy as e:
		yield _sse("error", {"message": str(e)})
	except (ValueError, DataSourceError) as e:
		yield _sse("error", {"message": str(e)})
	except Exception as e:
		yield _sse("error", {"message": f"{type(e).__name__}: {e}"})


@router.get("/data-source/auto-label/stream")
async def data_source_auto_label_stream(
	captcha_id: str = Query(...),
	rev: int = Query(1),
	device: str | None = Query(None),
	min_confidence: float = Query(0.0),
):
	"""draft 이미지를 모델 예측값으로 이름 바꾸는 진행 상황 SSE."""
	if not (0.0 <= min_confidence <= 1.0):
		raise HTTPException(status_code=400, detail="min_confidence 는 0 ~ 1 범위여야 합니다")
	if is_running():
		raise HTTPException(status_code=409, detail="이미 다른 수집/라벨링이 실행 중입니다")
	return TaskStreamingResponse(
		_auto_label_stream(captcha_id, rev, device, min_confidence),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
	)


@router.get("/data-source/confidence/stream")
async def data_source_confidence_stream(
	captcha_id: str = Query(...),
	rev: int = Query(1, ge=1),
	device: str | None = Query(None),
):
	"""DB 신뢰도가 없는 이미지에 대해서만 계산한다. 파일명을 변경하지 않는다."""
	if is_running():
		raise HTTPException(status_code=409, detail="이미 다른 수집/라벨링이 실행 중입니다")
	return TaskStreamingResponse(
		_auto_label_stream(captcha_id, rev, device, 0.0, rename_files=False),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
	)


@router.get("/data-source/drafts")
async def data_source_drafts(
	captcha_id: str = Query(...),
	rev: int = Query(1),
	limit: int | None = Query(None, ge=1),
):
	"""draft 에 쌓여 있는 이미지 목록. 갤러리 새로 고침이 이걸 읽는다.

	limit 을 주지 않으면 전부 돌려준다.
	"""
	return JSONResponse(list_drafts(captcha_id, rev, limit))


@router.post("/data-source/move-to-train")
def data_source_move_to_train(captcha_id: str = Query(...), rev: int = Query(1)):
	"""선택한 대상의 DB 확인 이미지를 학습 폴더로 이동한다."""
	try:
		return JSONResponse(move_checked_to_train(captcha_id, rev))
	except DataSourceBusy as e:
		raise HTTPException(status_code=409, detail=str(e))
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))


@router.post("/data-source/label")
async def data_source_label(
	captcha_id: str = Query(...),
	rev: int = Query(1),
	name: str = Query(...),
	label: str = Query(...),
):
	"""draft 라벨을 저장하고 수동 확인을 기록한다. 같은 라벨이면 파일명은 유지한다."""
	try:
		return JSONResponse(rename_draft(captcha_id, rev, name, label, manual_edit=True))
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))


@router.get("/data-source/image")
async def data_source_image(
	captcha_id: str = Query(...),
	rev: int = Query(1),
	name: str = Query(...),
):
	"""수집한 draft 이미지 썸네일."""
	try:
		path = draft_image_path(captcha_id, rev, name)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	return FileResponse(path, media_type="image/png")
