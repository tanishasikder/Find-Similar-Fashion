from contextlib import asynccontextmanager
import os
import sys
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
import mlflow
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import httpx

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from routers.input_router import router as input_router
from routers.db_router import router as db_router
from src.core.limiter import limiter

def database():
    supabase: Client = create_client(
        os.getenv('SUPABASE_URL'), # Initialize database
        os.getenv('SUPABASE_KEY')
    )
    bucket = supabase.storage.from_(os.getenv('BUCKET_NAME'))
    return bucket

app = FastAPI()
    
app.mount("/static", StaticFiles(directory="./"))

app.include_router(input_router) # For accepting user inputs
app.include_router(db_router) # For storing in the database

app.state.limiter = limiter # Initializes the rate limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)







    