from fastapi import APIRouter

from web.api.v1.batch import router as batch_router
from web.api.v1.data_source import router as data_source_router
from web.api.v1.predict import router as predict_router
from web.api.v1.train import router as train_router


router = APIRouter()
router.include_router(predict_router)
router.include_router(batch_router)
router.include_router(train_router)
router.include_router(data_source_router)
