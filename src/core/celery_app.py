from celery import Celery
from kombu import Queue
from fastapi import HTTPException, Request
import torch
from .config import Settings
from src.services.model_service.predict import image_output
from src.services.model_service.verify import image_preds
celery_app = Celery('fashionproject')

# No need to load from env when you do this
celery_app.config_from_object(Settings)

celery_app.task_routes = {
    'predict_img' : {'queue' : 'queue_image'}
}

@celery_app.task(name='predict-img', queue='queue_image') # Use the model and img from routers
def process_img(image_bytes: bytes):
    color, cat, attr = image_output(image_bytes) # Opens, preprocesses, and predicts
    true_color, true_cat, true_attr = image_preds(color, cat, attr)
    return true_color, true_cat, true_attr