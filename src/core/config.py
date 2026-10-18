import os
from dotenv import load_dotenv

# Load .env before reading REDIS_URL. celery_app imports this first, so without
# it a local run ignores .env and tries the docker hostname redis://redis
load_dotenv()

class Settings:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    broker_url = redis_url
    result_backend = redis_url
    task_serializer = "json"
    accept_content = ["json"]
    result_serializer = "json"
    timezone = "America/New_York"
    task_track_started = True
    task_time_limit = 300
    task_soft_time_limit = 250
    broker_connection_retry_on_startup = True  

settings = Settings()