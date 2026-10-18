from PIL import Image
from pathlib import Path
from multiprocessing import Pool
import os
import sys
import csv
import shutil
from collections import Counter
from dotenv import load_dotenv
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2] # Project root, where .env lives
sys.path.insert(0, str(BASE_DIR)) # So this works as a script and as an import

from src.process.process_color import hex_to_category

load_dotenv()

crop_images = BASE_DIR / os.getenv('CROPPED_IMAGES')
crop_csv = BASE_DIR / os.getenv('CROPPED_CSV', 'image_crop.csv')

# Columns in the csv: file name, color, category, attributes
FILE, COLOR = 0, 1

def label_crop(file_name):
    '''
    Recomputes the color of one saved crop. Returns None if the crop
    is missing or unreadable so the old label is kept
    '''
    try:
        with Image.open(crop_images / file_name) as img:
            return hex_to_category(img)
    except (OSError, ValueError):
        return None

def relabel():
    '''
    Rewrites only the color column of the crop csv using the current
    hex_to_category. Much faster than process_image.py since the crops
    are already saved, so nothing gets cropped again
    '''
    with open(crop_csv, 'r', newline='', encoding='utf-8') as f:
        rows = [row for row in csv.reader(f) if row]

    # The csv can have the same file more than once, only label each file once
    files = sorted({row[FILE] for row in rows})
    with Pool() as pool:
        colors = list(tqdm(
            pool.imap(label_crop, files, chunksize=256),
            total=len(files), desc='relabeling'
        ))
    new_color = {f: c for f, c in zip(files, colors) if c is not None}

    changed = 0
    for row in rows:
        color = new_color.get(row[FILE])
        if color and color != row[COLOR]:
            row[COLOR] = color
            changed += 1

    # Keep the old labels in case the new ones need to be compared or rolled back
    backup = crop_csv.with_name(crop_csv.name + '.bak')
    shutil.copy2(crop_csv, backup)
    # Write to a temp file first so a crash halfway never leaves a broken csv
    tmp = crop_csv.with_name(crop_csv.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    os.replace(tmp, crop_csv)

    print(f'Relabeled {len(new_color)} of {len(files)} crops, {changed} rows changed')
    print(f'Old labels saved to {backup}')
    counts = Counter(new_color.values())
    for color, count in counts.most_common():
        print(f'{color:7} {count / len(new_color):.3f}')

# Required for multi-processing in Windows
if __name__ == '__main__':
    relabel()
