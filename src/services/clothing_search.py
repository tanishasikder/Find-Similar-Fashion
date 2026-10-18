import base64
import io
import os
from functools import lru_cache
from typing import List
from channel3_sdk import Channel3
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MAX_SIDE = 1024 # Big enough for image search, keeps the request small

@lru_cache(maxsize=1)
def get_client():
    # Made on first use instead of at import so the app can still start without the key
    return Channel3(
        api_key=os.environ['CHANNEL3_API_KEY'],
        country='US',
        currency='USD',
    )

def shrink_image(contents: bytes):
    '''Resizes the upload and returns it as base64 jpeg (no data uri prefix)'''
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def search_products(contents: bytes, color: str, category: str, limit: int = 8) -> List[dict]:
    '''
    Searches Channel3 with the uploaded image plus the predicted color and
    category, and returns the in stock products with a link to buy them
    '''
    pager = get_client().products.search(
        query=f'{color} {category}',
        base64image=shrink_image(contents),
        limit=limit * 2, # Ask for extra since duplicates and products without offers get dropped
        filters={'availability': ['InStock']},
    )

    results, seen = [], set()
    # .items is only the first page, iterating the pager would fetch every page
    for product in pager.items or []:
        if not product.offers or product.title in seen:
            continue
        seen.add(product.title)

        offer = product.offers[0]
        images = product.images or []
        main = next((i for i in images if i.is_main_image), images[0] if images else None)

        results.append({
            'title': product.title,
            'brand': product.brands[0].name if product.brands else None,
            'url': offer.url,
            'store': offer.domain,
            'price': offer.price.price,
            'currency': offer.price.currency,
            'image': main.url if main else None,
        })
        if len(results) == limit:
            break

    return results
