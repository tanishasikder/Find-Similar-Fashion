from fastapi.responses import RedirectResponse
from src.schemas.input import ClothingRequest
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from postgrest.exceptions import APIError
import httpx
import logging

from src.schemas.db import supabase, SUPABASE_BUCKET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('supabase-app')

# Image is uploaded somewhere else the raw bytes are just passed here
def store_image(matrix: ClothingRequest,
                image: bytes = None,
                file_name: str = None):

    if image and file_name:
        image_filename = f"{matrix.color}_{matrix.category}_{file_name}"
        SUPABASE_BUCKET.upload(image_filename, image)

    supabase.table('clothes').insert({
        'color': matrix.color,
        'category': matrix.category,
        'attributes': matrix.attributes # Go on Supabase and change the tables
    }).execute()

    return RedirectResponse("/", status_code=303)

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True # Raise the real error after the last try, not a RetryError
)
def remove_expired():
    # Calculating the threshold of expiration. Supabase stores times in UTC
    time = datetime.now(timezone.utc) - timedelta(hours=2)
    # Turning time into the format Supabase wants
    exp = time.isoformat()

    try:
        response = (
            supabase.table('clothes')
            .delete()
            .lt('created_at', exp)
            .execute()
        )
        if not response.data:
            logger.warning(f'Delete matched 0 rows for exp={exp}')

        return response

    except APIError as e:
        # Postgrest/Supabase returned a structured error - bad filter, RLS violation
        logger.error(f'Supabase API Error: {e.message} | code={e.code} | details={e.details}')
        raise
    except httpx.TimeoutException:
        logger.error('Supabase request timed out')
        raise
    except httpx.ConnectError:
        logger.error('Could not reach Supabase - network/DNS issue')
        raise
