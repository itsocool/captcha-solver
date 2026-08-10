from fastapi import APIRouter

from web.api.v1.predict import router as predict_router


router = APIRouter()
router.include_router(predict_router)
