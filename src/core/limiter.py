from slowapi import Limiter
from slowapi.util import get_remote_address # Returns ip for current address
from dotenv import load_dotenv
import os

load_dotenv()
redis_url = os.getenv('REDIS_URL')

limiter = Limiter(key_func=get_remote_address, storage_uri=redis_url)
