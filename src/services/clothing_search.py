from channel3_sdk import Channel3
from typing import List

client = Channel3()

def search_products(features: List[str], link: str):
    color, cat, attr = features
    results = client.products.search(
        query = f'{color} clothing',
        filters = {
            'category' : cat,
            'attributes' : attr,
            'availability' : 'InStock'
        },
        image_url = link
    )

    return results

