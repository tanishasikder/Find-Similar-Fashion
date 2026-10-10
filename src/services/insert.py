from fastapi import APIRouter, Request, Depends, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from src.services.database import supabase, SUPABASE_BUCKET, SUPABASE_URL
from models import image_extraction
from fastapi.templating import Jinja2Templates
import os

'''
REMEMBER TO FIX THE FRONTEND YOU NEED TO GET THE USERS FILE NAME THEN
PUT IT EVERYWHERE ELSE AND THE COLOR, CAT, AND ATTR NEEDS TO BE IN A DICT
IN ANOTHER PLACE YOU HAVE IT AS A LIST AND OTHER PLACES ITS SEPARATE
VARIABLES SO PUT IT ALL IN A DICT OR CHANGE TO WHATEVERS BEST PRACTICE
'''

def add_clothes(
    file_content : bytes,
    file_name : str,
    clothes : image_extraction
):
    image_url = None
    if file_content and file_name != "":
        image_filename = f"{clothes.color}_{clothes.category}_{file_name}"
        response = supabase.storage.from_(SUPABASE_BUCKET).upload(image_filename, file_content)
        if response.status_code == 200:
            image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filename}"

    supabase.table('clothes').insert({
        'color': clothes.color,
        'category': clothes.category,
        'attributes' : clothes.attributes # Go on supabase and add this
    }).execute()

    return RedirectResponse("/", status_code=303)

def get_cloth_link(clothes : Dict, file_name: str):
    image_filename = f"{clothes.color}_{clothes.category}_{file_name}"
    image_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filename}"

    return image_url

