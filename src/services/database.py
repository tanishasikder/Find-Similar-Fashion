from fastapi.responses import RedirectResponse
from supabase import create_client, Client
import os
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from PIL import Image
from src.schemas.input import ClothingRequest
from typing import Optional, List
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from postgrest.exceptions import APIError
import httpx
import logging
import sys 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.schemas.db import supabase, SUPABASE_BUCKET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('supabase-app')

# Image is uploaded somewhere else the raw bytes are just passed here
def store_image(matrix: List[ClothingRequest],
                image: bytes = None,
                SUPABASE_URL = None, 
                supabase : Client = None):
    
    image_url = None
    if image and image.filename != "":
        image_filename = f"{matrix.color}_{matrix.category}_{image.filename}"
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(image_filename, image)
        if response.status_code == 200:
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filename}"

    supabase.table('clothes').insert({
        'color': matrix.color,
        'category': matrix.category,
        'attributes': matrix.attributes # Go on Supabase and change the tables
    }).execute()

    return RedirectResponse("/", status_code=303)

@retry(
    retry=retry_if_exception_type((httpx.TimeOutException, httpx.ConnectError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8)
)

def remove_expired():
    # Calculating the threshold of expiration
    time = datetime.now() - timedelta(hours=2)
    # Turning time into the format Supabase wants
    exp = time.isoformat()

    try:
        response = (
            supabase.table('clothes')
            .delete()
            .select('created_at')
            .lt('created_at', exp)
            .execute()
        )
        if not response.data:
            logger.warning(f'Delete matched 0 rows for exp={exp}')

        return response

    except APIError as e:
        # Postgrest/Supabase returned a structured error — bad filter, RLS violation
        logger.error(f'Supabase API Error: {e.message} | code={e.code} | details={e.details}')
        raise
    except httpx.TimeoutException:
        logger.error('Supabase request timed out')
        raise
    except httpx.ConnectError:
        logger.error('Could not reacj Supabase - network/DNS issue')
        raise