import base64
import logging
from typing import List
from celery.result import AsyncResult
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import Request, status
from fastapi import Depends, Form
from fastapi.concurrency import run_in_threadpool
from PIL import UnidentifiedImageError
from pydantic import BaseModel
from src.core.limiter import limiter
from src.core.celery_app import celery_app, process_img
from src.schemas.jwt import verify_jwt
from src.services.clothing_search import search_products
from src.services.predict import predict_image
from src.services.rag import build_query, get_rag_response

logger = logging.getLogger(__name__)

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
    user_id = payload.get('sub')
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

# Same shape as the dict services/predict.py returns
class Prediction(BaseModel):
    color: str
    category: str
    attributes: List[str] = []

# The frontend calls this after /status gives back the image prediction.
# The RAG agent describes the clothing and how it fits recent trends
@router.post("/query")
@limiter.limit('3/minute')
async def get_query_rag(
        request : Request,
        prediction: Prediction,
        payload: dict = Depends(verify_jwt) # Each call costs Gemini/Pinecone usage so require login
    ):
    try:
        answer = await get_rag_response(build_query(prediction.model_dump()))
    except Exception:
        logger.exception('RAG request failed')
        raise HTTPException(status_code=503, detail='Could not get fashion insights right now')
    return {'answer': answer}

# The frontend calls this after /status gives back the image prediction.
# Searches stores with the same image plus the predicted color and category
@router.post("/similar")
@limiter.limit('3/minute')
async def get_similar_clothes(
        request : Request,
        file: UploadFile = File(...),
        color: str = Form(...),
        category: str = Form(...),
        payload: dict = Depends(verify_jwt) # Each call costs Channel3 usage so require login
    ):
    contents = await file.read()
    try:
        # The Channel3 client blocks so run it off the event loop
        products = await run_in_threadpool(search_products, contents, color, category)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail='File is not a valid image')
    except Exception:
        logger.exception('Similar clothes search failed')
        raise HTTPException(status_code=503, detail='Could not find similar clothes right now')
    return {'products': products}

