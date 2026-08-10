from fastapi import APIRouter
from fastapi.responses import JSONResponse

from web.core.db import get_service_config
from web.core.version import get_app_version
from web.services.captcha import loaded_model_ids


router = APIRouter()


@router.get("/health")
async def health():
	config = get_service_config()
	loaded = loaded_model_ids()
	serviced = config["serviced"]
	return JSONResponse({
		"status": "ok" if set(serviced) <= set(loaded) else "degraded",
		"version": get_app_version(),
		"default_captcha_id": config["default_captcha_id"],
		"serviced_captcha_ids": serviced,
		"loaded_captcha_ids": loaded,
		"config_source": config["source"],
	})


@router.get("/ping")
async def ping():
	return JSONResponse({"ping": "pong"})


@router.get("/version")
async def version():
	return JSONResponse({"version": get_app_version()})
