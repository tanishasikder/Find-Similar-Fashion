import base64
from celery import Celery
from celery.signals import worker_process_init
from .config import Settings
from src.schemas.state import load_image_model
from src.services.predict import predict_image
celery_app = Celery('fashionproject')

# No need to load from env when you do this
celery_app.config_from_object(Settings)

celery_app.conf.task_routes = {
    'predict-img' : {'queue' : 'queue_image'}
}

# The worker is a separate process from the api so it can't see app.state.
# Each worker process loads the model once and reuses it for every task
image_model = None

def get_model():
    global image_model
    if image_model is None:
        image_model = load_image_model()
    return image_model

@worker_process_init.connect
def load_model(**kwargs):
    get_model()

@celery_app.task(name='predict-img', queue='queue_image')
def process_img(image_b64: str):
    # Task args are sent as json, so the image comes in base64 encoded
    contents = base64.b64decode(image_b64)
    return predict_image(get_model(), contents) # Opens, preprocesses, and predicts
