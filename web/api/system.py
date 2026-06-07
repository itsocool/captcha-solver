from fastapi import APIRouter
from fastapi.responses import JSONResponse

from web.core.version import get_app_version


router = APIRouter()


@router.get("/health")
async def health():
	return JSONResponse({"status": "ok"})


@router.get("/ping")
async def ping():
	return JSONResponse({"ping": "pong"})


@router.get("/version")
async def version():
	return JSONResponse({"version": get_app_version()})
