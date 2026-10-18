from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import httpx
from src.core.limiter import limiter
from src.core.tracking_config import Dagshub_Track
from src.routers.input_router import router as input_router
from src.routers.db_router import router as db_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything here loads once at startup and is shared by every request
    tracker = Dagshub_Track() # Use the DagsHub Mlflow server to log things
    tracker.initialize()
    app.state.tracker = tracker

    yield # Let the app process

    await app.state.http_client.aclose()

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="Frontend"))

# The homepage is the upload page in Frontend/index.html
@app.get("/", include_in_schema=False)
def home():
    return FileResponse("Frontend/index.html")

app.include_router(input_router) # For accepting user inputs
app.include_router(db_router) # For storing in the database

app.state.limiter = limiter # Initializes the rate limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
