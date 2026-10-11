from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address # Returns ip for current address
from dotenv import load_dotenv
import os

load_dotenv()
redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0') # Same default as core/config.py

# Behind nginx every request comes from the proxy's ip, so use the client
# ip nginx forwards in X-Real-IP or every user would share one limit
def get_client_ip(request: Request):
    return request.headers.get('X-Real-IP') or get_remote_address(request)

limiter = Limiter(key_func=get_client_ip, storage_uri=redis_url)
