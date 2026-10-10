from dotenv import load_dotenv
import os 
from pathlib import Path
import numpy as np
import pandas as pd
import json
from collections import defaultdict

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2] # Project root, where .env lives

annotations = BASE_DIR / os.getenv('ANNOTATION_DIR')
path = annotations / 'train_annotations.json'


def get_images():
    '''
    Goes through every image and gets IDs of categories and 
    attributes. Stores everything in a dict
    '''
    images_temp = [] # Temporary hold all dicts before processing
    file_names = []

    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)

    # They are just IDs so need to decode them later on
    for annotation in file['annotations']:
        values = {'image_id' : annotation['image_id'],
                  'attribute_id' : annotation['attribute_ids'],
                  'category_id' : annotation['category_id'],
                  'bbox' : annotation['bbox']}

        images_temp.append(values)

    for image in file['images']:
        # Get file name and ID to match the other list with
        file_names.append({'name' : image['file_name'],
                           'id' : image['id']})

    return file_names, images_temp

def process_values(file_names):
    images = {}
    for value in file_names:
        images[value['name']] = value
        
    return images

def get_cat(id, cats):
    cat = next(c['name'] for c in cats if c['id'] == id)
    return cat

def get_attr(attr_id, attrs):
    names = []
    
    for id in attr_id:
        names.append(next(a['name'] for a in attrs if a['id'] == id))

    return names

def decode_images(images, images_temp):
    processed = {}

    # Decodes all of the IDs for each image
    with open(path, 'r', encoding='utf-8') as t:
        file = json.load(t)

    # Each image has 0 or more attributes in a list
    # Each image has only one category
    categories = file['categories']
    attributes = file['attributes']
    # Index annotations by image_id 
    by_image_id = defaultdict(list)
    for item in images_temp:
        by_image_id[item['image_id']].append(item)

    for img in images:
        id = images[img]['id']
        # Get the dict in images_temp that has the values first
        results = by_image_id[id]
        for result in results:
            cat = get_cat(result['category_id'], categories)
            describe = result['attribute_id']
            bbox = result['bbox'] # No need to process this. Used to find stuff later

            if describe: # Not every img has an attribute
                attr = get_attr(describe, attributes)
            else:
                attr = 'None'

            # Every image is a list of [cat, attr, bbox], one per object
            processed.setdefault(img, []).append([cat, attr, bbox])

    return processed

