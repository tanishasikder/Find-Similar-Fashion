import io
import torch
from PIL import Image
from src.models.image_extraction import device
from src.process.transform import eval_transform
from src.schemas.state import image_preds

transform = eval_transform()

# Gets the model predictions for color, clothing type, and attributes
def predict_image(model, contents: bytes):
    opened = Image.open(io.BytesIO(contents)).convert('RGB')
    # Same preprocessing as testing, plus a batch dimension
    batch = transform(opened).unsqueeze(0).to(device)
    # Perform inference
    with torch.no_grad():
        color, cat, attr = model(batch)

    return image_preds(model, color, cat, attr)
