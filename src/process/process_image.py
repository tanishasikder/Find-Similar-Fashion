from PIL import Image
from pathlib import Path
import os
import sys
import json
import csv
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir)) # So this works as a script and as an import

from src.process.process_color import hex_to_category

load_dotenv()

categories = os.environ.get('TYPE_LABEL')
cloth_labels = os.environ.get('FASHION_LABELS')
cloth_images = os.environ.get('IMAGE_FASHION_DIR')
crop_images = os.environ.get('CROPPED_IMAGES')
crop_csv = os.environ.get('CROPPED_CSV', 'image_crop.csv')

def get_type_labels():
    objects = []
    detailed = []

    for d in os.listdir(categories):
        with open(os.path.join(categories, d), 'r', encoding='utf-8') as f:
            if d == 'objects.txt':
                objects.extend(f.read().splitlines())
            elif d == 'fine_details.txt':
                detailed.extend(f.read().splitlines())

    return objects, detailed

def image_labels():
    with open(cloth_labels, 'r') as f:
        labels = json.load(f)

    return labels

def extract_labels(labels, file):
    '''
    Each image is stored as [cat, attr, bbox, [cat, attr, bbox], ...].
    The first object is flattened into the list so split it back out
    to get a list of [cat, attr, bbox] for every object.
    '''
    values = labels.get(file)
    if not values:
        return []
    if isinstance(values[0], str):
        return [values[:3]] + values[3:]
    return values

def get_data(values, file, folder, writer, done):
    with open(os.path.join(folder, file), 'rb') as f:
        img = Image.open(f).convert('RGB')

    for val in values:
        if len(val) != 3 or not isinstance(val[-1], list):
            continue # Skip if things are wrong.

        crop, dimen = crop_image(img, val[-1])
        if crop is None:
            continue

        # Need to make the filenames unique so use dimen and separate with _
        # Having the dimensions makes the filenames always the same if you
        # Run the code again
        id = "".join(dimen)
        file_name = f'{id}_{file}'
        if file_name in done:
            continue
        done.add(file_name)

        color = hex_to_category(crop) # Used to classify image color
        cat = val[-3]
        attr = val[-2]

        crop.save(Path(crop_images) / file_name)
        writer.writerow([file_name, color, cat, attr])

def crop_image(img, values):
    if len(values) < 4:
        return None, None

    x, y, w, h = values # Fashionpedia does not follow PIL format

    if w <= 0 or h <= 0:
        return None, None

    dimen = [x, y, x + w, y + h] # left, top, right, bottom
    crop = img.crop(dimen)
    # Return the dimension to label the file later on
    return crop, [str(v) for v in dimen]

def pass_images():
    labels = image_labels() # Mapping of file -> categories, attributes
    Path(crop_images).mkdir(parents=True, exist_ok=True)
    Path(crop_csv).parent.mkdir(parents=True, exist_ok=True)

    # Rows already written are skipped so the csv never gets duplicates
    # and a stopped run can pick back up where it left off
    done = set()
    if os.path.exists(crop_csv):
        with open(crop_csv, 'r', newline='', encoding='utf-8') as f:
            done = {row[0] for row in csv.reader(f) if row}

    with open(crop_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for folder, _, files in os.walk(cloth_images):
            for file in files:
                values = extract_labels(labels, file)
                if values:
                    get_data(values, file, folder, writer, done)

if __name__ == '__main__':
    pass_images()
