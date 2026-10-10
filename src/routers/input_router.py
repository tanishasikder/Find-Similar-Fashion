import base64
from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import Request, status
from fastapi import Depends
from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError
from src.core.limiter import limiter
from src.core.celery_app import celery_app, process_img
from src.schemas.jwt import verify_jwt
from src.services.predict import predict_image

# No prefix since the frontend calls /upload and /status directly
router = APIRouter()

# Basic health check to ensure server is functioning
@router.get("/health")
def root():
    return {"status" : "OK"}

@router.post("/upload")
@limiter.limit('3/minute') # How much we limit
async def upload(
        request : Request, # Need this or limiter will not work
        payload: dict = Depends(verify_jwt), # Get this from schemas/jwt.py it verifies the user using jwt
        file: UploadFile = File(...)
    ):
    user_id = payload['sub']
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Authorization is missing',
            headers={'WWW-Authenticate': 'Bearer'}
        )
    contents = await file.read()
    # Celery sends task args as json, which can't hold raw bytes
    encoded = base64.b64encode(contents).decode('ascii')
    task = process_img.delay(encoded) # Process this image in the worker
    # The prediction runs in the background, poll /status/{task_id} for it
    return {'task_id': task.id}

# The frontend polls this until the celery task finishes
@router.get('/status/{task_id}')
def task_status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    if result.successful():
        return {'status': 'SUCCESS', 'result': result.result}
    if result.failed():
        return {'status': 'FAILURE', 'error': str(result.result)}
    return {'status': result.status} # PENDING or STARTED

# Use with services/rag function get_rag_response
@router.get("/query")
@limiter.limit('3/minute')
async def get_query_rag(request : Request, query: str):
    return query

# Gets the model predictions right away using the model loaded at startup, no celery
@router.post('/image_predict')
@limiter.limit('3/minute')
async def get_image_preds(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    try:
        # Inference blocks so run it off the event loop
        return await run_in_threadpool(predict_image, request.app.state.image_model, contents)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail='File is not a valid image')
