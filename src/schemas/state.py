from contextlib import asynccontextmanager
import os
import sys
from fastapi import FastAPI
import torch
from torchvision import transforms
import torchvision.models as models
# Loading in the custom model
from pathlib import Path
import sys
from joblib import load
from PIL import Image
import json
from dotenv import load_dotenv
import os
# Makes python looks at the parent root directories to find the model
load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2] # Project root, where .env lives

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = BASE_DIR / os.getenv('CLASS_LABELS')
cnn = BASE_DIR / os.getenv('IMAGE_MODEL')

# Loops through the file names and stores all colors and categories
def classes():
    with open(class_names, "r") as f:
        labels = json.load(f)

    co = labels["color"]
    ca = labels["category"]
    attr = labels["attribute"]

    return co, ca, attr

def initialize_image_model(): 
    co, ca, attr = classes()
    # Loading in the clothing predict model with error handling 
    image_model = cnn(co, ca, attr)
    # Loading in custom weights
    torch_path = BASE_DIR / "image_extraction_model.pth"
    image_model.load_state_dict(torch.load(torch_path, map_location=device))
    image_model.eval()

    return image_model

def image_preds(color_pred, cat_pred, attr_pred):
    '''
    Get the english words for the color and category.
    This is used after prediction
    '''
    # We need this to find the true english labels
    colors, cats, attrs = classes()

    color_arg = torch.argmax(color_pred, dim=1)
    cat_arg = torch.argmax(cat_pred, dim=1)
    attr_arg = torch.argmax(attr_pred, dim=1)

    # Find the color and category based on the gotten index
    color, cat, attr = colors[color_arg], cats[cat_arg], attrs[attr_arg]

    return color, cat, attr