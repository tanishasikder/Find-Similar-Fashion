import os
import json
from pathlib import Path
import torch
from dotenv import load_dotenv

from src.models.image_extraction import CNN, device

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2] # Project root, where .env lives

class_names = BASE_DIR / os.getenv('CLASS_LABELS')
model_dir = BASE_DIR / os.getenv('IMAGE_MODEL')

# Loads the colors, categories, and attributes in the order of the model outputs
def classes():
    with open(class_names, "r", encoding="utf-8") as f:
        labels = json.load(f)

    co = labels["color"]
    ca = labels["category"]
    attr = labels["attribute"]

    return co, ca, attr

def load_image_model():
    '''
    Builds the CNN and loads the trained weights. Call this once at
    startup and reuse the model, it is too big to load per request
    '''
    co, ca, attr = classes()
    # No pretrained download since the trained weights replace them anyway
    image_model = CNN(co, ca, attr, pretrained=False)
    torch_path = model_dir / "image_extraction_model.pth"
    image_model.load_state_dict(torch.load(torch_path, map_location=device, weights_only=True))
    image_model.eval()

    return image_model

def image_preds(model, color_pred, cat_pred, attr_pred, threshold=0.5):
    '''
    Get the english words for the color, category, and attributes.
    This is used after prediction on a batch of one image
    '''
    color = model.color_names[color_pred.argmax(1).item()]
    cat = model.category_names[cat_pred.argmax(1).item()]
    # Attributes are multi label so every one over the threshold is on
    attr_on = (torch.sigmoid(attr_pred[0]) > threshold).nonzero().flatten().tolist()
    attrs = [model.apparels[i] for i in attr_on]

    return {'color': color, 'category': cat, 'attributes': attrs}
