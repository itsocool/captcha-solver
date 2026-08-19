from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from web.core.config import get_settings
from web.core.db import get_service_config
from web.core.device import device_options
from web.core.version import get_app_version
from web.services.batch_predict import list_targets
from web.services.captcha import list_captcha_types, model_status
from web.services.data_source import MAX_COUNT, MAX_DELAY_MS, list_targets as data_source_targets
from web.services.train import LOSS_TYPES, PARAM_SPEC, list_targets as train_targets


def create_router(templates: Jinja2Templates) -> APIRouter:
	router = APIRouter()

	@router.get("/", response_class=HTMLResponse)
	async def index(request: Request):
		config = get_service_config()
		return templates.TemplateResponse(
			request=request,
			name="index.html",
			context={
				"active_nav": "predict",
				"default_captcha_id": config["default_captcha_id"],
				"captcha_types": list_captcha_types(),
				"device_options": device_options(),
				"app_version": get_app_version(),
				"predict_image_url": f"{get_settings().web_context_path}/api/v1/predictImage",
			},
		)

	@router.get("/predict", response_class=HTMLResponse)
	async def batch_predict(request: Request):
		return templates.TemplateResponse(
			request=request,
			name="predict.html",
			context={
				"active_nav": "batch_predict",
				"targets": list_targets(),
				"device_options": device_options(),
				"app_version": get_app_version(),
			},
		)

	@router.get("/train", response_class=HTMLResponse)
	async def train(request: Request):
		return templates.TemplateResponse(
			request=request,
			name="train.html",
			context={
				"active_nav": "train",
				"targets": train_targets(),
				"device_options": device_options(),
				"param_spec": PARAM_SPEC,
				"loss_types": LOSS_TYPES,
				"app_version": get_app_version(),
			},
		)

	@router.get("/data-source", response_class=HTMLResponse)
	async def data_source(request: Request):
		return templates.TemplateResponse(
			request=request,
			name="data_source.html",
			context={
				"active_nav": "data_source",
				"targets": data_source_targets(),
				"max_count": MAX_COUNT,
				"max_delay_ms": MAX_DELAY_MS,
				"app_version": get_app_version(),
			},
		)

	@router.get("/status", response_class=HTMLResponse)
	async def status(request: Request):
		config = get_service_config()
		models = model_status()
		devices = {model["device"] for model in models if model["loaded"]}
		return templates.TemplateResponse(
			request=request,
			name="status.html",
			context={
				"active_nav": "status",
				"models": models,
				"device": ", ".join(sorted(devices)) if devices else "-",
				"default_captcha_id": config["default_captcha_id"],
				"config_source": config["source"],
				"app_version": get_app_version(),
			},
		)

	return router
