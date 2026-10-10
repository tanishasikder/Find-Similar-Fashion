from fastapi import Request, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from src.services.database import supabase
from models import image_extraction
from fastapi.templating import Jinja2Templates
import os
from src.routers.input_router import router
from src.main import limiter 
from fastapi import APIRouter, Request
from src.services.insert import add_clothes
router = APIRouter()

TEMPLATE_PATH=os.getenv('TEMPLATE_PATH')

templates = Jinja2Templates(directory=TEMPLATE_PATH)

@router.get("/", response_class=HTMLResponse)
@limiter.limit('3/minute')
async def read_clothes(request: Request): 
    '''
    Use in services/db_service/ insert function 
    read_clothes, add_clothes_form
    '''
    return request

@router.post('/add')
@limiter.limit('3/minute')
async def get_clothes( 
    request : Request, # Need this or limiter will not work
    clothes : image_extraction = Depends(image_extraction.as_form),
    image: UploadFile = File(None), 
):
    '''
    Use for services/db_service/insert 
    function add_clothes.

    Clothes is used by Depends(image_extraction.as_form), 
    which gives you a Pydantic model instance from form 
    data the user submitted
    '''
    if image and image.filename != "":
        file = image.filename
        content = await image.read()

        add_clothes(file, content, clothes)

def read_clothes(request: Request):
    response = supabase.table('clothes').select('*').eq('is_active', True).execute()
    clothes = response.data
    return templates.TemplateResponse('info.html', {'request': request, 'clothes': clothes})

def add_clothes_form(request: Request):
    return templates.TemplateResponse('add_clothes.html', {'request': request})

        
