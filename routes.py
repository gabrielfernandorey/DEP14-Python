from fastapi import APIRouter
from controllers.task_controller import router as task_router

router = APIRouter()

router.include_router(task_router)