from PIL import Image
from pathlib import Path
import os
import copy
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import Dataset
import torch
import ast
from sklearn.preprocessing import MultiLabelBinarizer

from src.process.process_color import COLOR_NAMES, get_colors
from src.process.transform import fashion_transform

load_dotenv()

cropped = os.environ.get('CROPPED_IMAGES')
names = os.environ.get('CROPPED_CSV')

# Columns in the csv: file name, color, category, attributes
FILE, COLOR, CATEGORY, ATTR = 0, 1, 2, 3

def clean(df):
    '''
    Replace all attributes with '' if none else leave it alone
    '''
    df.iloc[:, ATTR] = df.iloc[:, ATTR].fillna('')
    return df

df = pd.read_csv(names, header=None, keep_default_na=False)
df = df.drop_duplicates(subset=FILE).sort_values(by=df.columns[FILE])
data = clean(df)

def to_list(s):
    if s is None or s == '' or s == 'None':
        return []
    val = ast.literal_eval(s)
    return val if isinstance(val, list) else []

# Colors use a fixed mapping so the indices match the model's color head
color_codes = data.iloc[:, COLOR].map(get_colors()).to_numpy()
cat_codes, cat_classes = pd.factorize(data.iloc[:, CATEGORY]) # Get all the categories and attributes
# Then encode and return as a list
remove = data.iloc[:, ATTR].apply(to_list)
encode = MultiLabelBinarizer()
attr = encode.fit_transform(remove)

def get_label_classes(encoder):
    # The labels are encoded so this makes a mapping of the decoded -> encoded
    mappings = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    return mappings

def get_class_names():
    # Names for each output of the model in the same order as the labels
    return list(COLOR_NAMES), list(cat_classes), list(encode.classes_)

class ImageData(Dataset):
    def __init__(self, dir=cropped, transform=None):
        self.dir = Path(dir)
        self.transform = transform if transform is not None else fashion_transform()
        # Labels stay as small numpy arrays and only become tensors in __getitem__.
        # Windows copies the dataset into every dataloader worker, and a dict of
        # ~1 million tiny tensors takes minutes per worker to copy
        rows = {fname: i for i, fname in enumerate(data.iloc[:, FILE])}
        self.image_paths = sorted([
            path.name for path in self.dir.iterdir()
            if path.name in rows
        ]) # Loop through all images
        order = np.array([rows[p] for p in self.image_paths], dtype=np.int64)
        self.colors = color_codes[order].astype(np.int64)
        self.cats = cat_codes[order].astype(np.int64)
        self.attrs = attr[order].astype(np.uint8)

    def source_ids(self):
        # Crops are named <box>_<source image>.jpg so crops of one photo share the suffix
        return [p.split('_', 1)[-1] for p in self.image_paths]

    def subset(self, indices):
        # Copy that only holds the given rows. Unlike torch Subset, the dataloader
        # workers then only get this split's labels pickled to them, not the whole dataset
        part = copy.copy(self)
        part.image_paths = [self.image_paths[i] for i in indices]
        part.colors = self.colors[indices]
        part.cats = self.cats[indices]
        part.attrs = self.attrs[indices]
        return part

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        color = torch.tensor(self.colors[idx], dtype=torch.long)
        cat = torch.tensor(self.cats[idx], dtype=torch.long)
        attr = torch.tensor(self.attrs[idx], dtype=torch.float32)
        image = Image.open(self.dir / path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, color, cat, attr

if __name__ == "__main__":
    print(data.iloc[:, FILE].tolist())
    print(data.index)
