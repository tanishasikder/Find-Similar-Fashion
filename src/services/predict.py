import torch
from PIL import Image
from fastapi import HTTPException
from src.schemas.state import ImageService
from src.schemas.state import initialize_image_model
import io

# Gets the model predictions for color and clothing type
async def image_output(contents: bytes):
    try:
        opened = Image.open(io.BytesIO(contents))
        # Get the image model put into app
        image_model = initialize_image_model(ImageService)
        # Perform inference
        with torch.no_grad():
            color, cat, attr = image_model(opened)
        
        return color, cat, attr
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")



