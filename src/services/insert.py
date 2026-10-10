from fastapi.responses import RedirectResponse
from src.schemas.db import supabase, SUPABASE_BUCKET

'''
REMEMBER TO FIX THE FRONTEND YOU NEED TO GET THE USERS FILE NAME THEN
PUT IT EVERYWHERE ELSE AND THE COLOR, CAT, AND ATTR NEEDS TO BE IN A DICT
IN ANOTHER PLACE YOU HAVE IT AS A LIST AND OTHER PLACES ITS SEPARATE
VARIABLES SO PUT IT ALL IN A DICT OR CHANGE TO WHATEVERS BEST PRACTICE
'''

def cloth_filename(clothes : dict, file_name : str):
    # Prefix with the prediction so images are easy to find in the bucket
    return f"{clothes['color']}_{clothes['category']}_{file_name}"

def add_clothes(
    file_content : bytes | None,
    file_name : str | None,
    clothes : dict
):
    if file_content and file_name:
        # Raises if the upload fails so nothing half saved gets inserted
        SUPABASE_BUCKET.upload(cloth_filename(clothes, file_name), file_content)

    supabase.table('clothes').insert({
        'color': clothes['color'],
        'category': clothes['category'],
        'attributes' : clothes['attributes'] # Go on supabase and add this
    }).execute()

    return RedirectResponse("/", status_code=303)

def get_cloth_link(clothes : dict, file_name: str):
    return SUPABASE_BUCKET.get_public_url(cloth_filename(clothes, file_name))
