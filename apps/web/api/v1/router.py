from fastapi import APIRouter

from web.api.v1.batch import router as batch_router
from web.api.v1.predict import router as predict_router


router = APIRouter()
router.include_router(predict_router)
router.include_router(batch_router)
