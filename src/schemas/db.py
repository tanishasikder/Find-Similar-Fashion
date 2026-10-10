from supabase import create_client, Client
import os

SUPABASE_URL = os.environ.get('SUPABASE_URL')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

SUPABASE_BUCKET = supabase.storage.from_(BUCKET_NAME)
