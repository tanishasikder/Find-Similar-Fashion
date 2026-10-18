from typing import List
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from src.core.limiter import limiter
from src.schemas.db import supabase
from src.services.insert import add_clothes
router = APIRouter()

TEMPLATE_PATH=os.getenv('TEMPLATE_PATH', 'templates')

templates = Jinja2Templates(directory=TEMPLATE_PATH)

# Not "/" since that is the upload page (see main.py)
@router.get("/clothes", response_class=HTMLResponse)
@limiter.limit('3/minute')
def read_clothes(request: Request):
    # Shows every active piece of clothing in the database
    response = supabase.table('clothes').select('*').eq('is_active', True).execute()
    clothes = response.data
    return templates.TemplateResponse(request, 'info.html', {'clothes': clothes})

@router.get('/add', response_class=HTMLResponse)
def add_clothes_form(request: Request):
    return templates.TemplateResponse(request, 'add_clothes.html')

@router.post('/add')
@limiter.limit('3/minute')
async def get_clothes(
    request : Request, # Need this or limiter will not work
    color: str = Form(...),
    category: str = Form(...),
    attributes: List[str] = Form([]),
    image: UploadFile = File(None),
):
    '''
    Stores the clothing (and its image if one was given) using
    services/insert function add_clothes, then redirects to /clothes
    '''
    # The form has a few attribute boxes, so drop the ones left empty
    attributes = [a.strip() for a in attributes if a.strip()]
    clothes = {'color': color, 'category': category, 'attributes': attributes}
    file, content = None, None
    if image and image.filename != "":
        file = image.filename
        content = await image.read()

    # Supabase calls block so run them off the event loop
    return await run_in_threadpool(add_clothes, content, file, clothes)
