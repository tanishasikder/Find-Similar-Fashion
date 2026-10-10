from datetime import datetime, timedelta
import os
import uuid
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import io
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys
from joblib import load
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

'''
GET RID OF IMAGE MODEL IN image_model_output
'''

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.api.services.rag import get_rag_response
router = APIRouter()

# Loading in the custom model
from src.models.image_extraction import CNN
from src.models.training.sales_predict import MTGBM

# Makes python looks at the parent root directories to find the model
parent = Path(__file__).parent
path = parent / "stats_model.joblib"
stats_model = load(path)
app = FastAPI()
app.include_router(router)

# Basic health check to ensure server is functioning
@app.get("/health")
def root():
    return {"status" : "OK"}

# Function that initializes the image model when app starts
@app.on_event("startup")
async def startup_event():
    return initialize_image_model()

# Temporary storage for image features
# Store image_id, image characteristics, when uploaded
# (so it can expire after an hour) for predictions
image_storage = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clothingpredict.com"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"]
)

# Initial parameters to predict with later on
class ClothingParameters(BaseModel):
    size : str = None
    catalog_price : float = None
    channel : str = None
    original_price : float = None

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Transformations for user input images
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Loops through the file names and stores all colors and categories
def get_color_category():
    path = 'Fashion_Images/train'
    files = os.listdir(path)
    color = [file[:file.index('_')] for file in files]
    category = [file[file.index('_')+1:] for file in files]

    return color, category

def initialize_image_model():
    color, category = get_color_category()
    # Loading in the clothing predict model with error handling 
    image_model = CNN(color, category)
    # Loading in custom weights
    torch_path = parent / "image_extraction_model.pth"
    image_model.load_state_dict(torch.load(torch_path, map_location='cpu'))
    image_model.eval()

    return image_model

# Gets the model predictions for color and clothing type
async def image_model_output(file: UploadFile) -> Image.Image:
    # Make sure the file is an image
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image. Try again")
    # Wait for the file to be read
    contents = await file.read()
    try:
        remove_expired_images()
        # Get rid of this. You need to call the start function
        #then use the returned value as the image model
        #not initialize it here
        image_model = initialize_image_model()
        # Extract the image from content
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        # Create image ID
        image_id = str(uuid.uuid4())
        image = data_transforms(image).unsqueeze(0)
        # Perform inference
        with torch.no_grad():
            color, cloth_type = image_model(image)

        image_storage[image_id] = {
            'features': [color, cloth_type],
            'uploaded': datetime.now()
        }
        return color, cloth_type
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")
    
def remove_expired_images():
    # Remove images after an hour
    for key, data in list(image_storage.items()):
        upload = image_storage[key]['uploaded']
        if datetime.now() >= upload + timedelta(hours=1):
            image_storage.pop(key)

# Defines dictionary for acceptable ranges and types
def acceptable_values():
    template = {
        'size' : {'values': { 'XS', 'S', 'M', 'L', 'XL'}, 
                  'types': str},
        'catalog price' : {'values': range(5, 1000),
                           'types': (int, float)},
        'channel' : {'values': {'App Mobile', 'E-Commerce'}, 
                     'types': str},
        'original price' : {'values': range(5, 1000), 
                            'types': (int, float)}
    }

    return template

# Checks if theres an empty field
@field_validator()
@classmethod
def verify_inputs(user_param):
    cleaned = user_param.strip()

    # If the string is empty
    if not cleaned:
        raise ValueError("Field cannot be empty")

    return cleaned
    
# Checks if the user typed in acceptable parameters
# Then combines if its fine
#@app.post("/files")
async def get_user_params(
    # Select is the task the user wants
    matrix: List[ClothingParameters],
    select : int,
    color : str, 
    category : str
):
    try:
        numerical_outputs = []
        template = acceptable_values()
        # Combine the input parameters with calculated values
        for data in matrix:
            verify_inputs(data)
            user_params = {
                'size' : data.size,
                'catalog price' : data.catalog_price,
                'channel' : data.channel,
                'original price' : data.original_price
            }

            if user_params[data] not in template[data]:
                return "Not OK"
            if not isinstance(user_params[data], template[data]['types']):
                return "Not OK"
            # Predict with the other model
            result = stats_model.clothing_predict([color, category, user_params, select])
            numerical_outputs.append(result.tolist())
        return numerical_outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail="Sorry. Prediction Failed")

@router.get("/query/")
async def query_rag_system(query: str):
    try:
        response = await get_rag_response(query)
        return {"query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#@app.post("/predict")
async def initialize_preds(
    # Select is the task the user wants
    matrix: List[ClothingParameters], 
    file : UploadFile = File(...), 
    select : int = Body(...)
):
    # We need this to find the true english labels
    color_labels, category_labels = get_color_category()

    color_py, category_py = await image_model_output(file)
    color_pred = torch.argmax(color_py, dim=1)
    cat_pred = torch.argmax(category_py, dim=1)
    
    # Find the color and category based on the gotten index
    color, category = color_labels[color_pred], category_labels[cat_pred]
    # Verifies if user parameters match the requirements
    if verify_inputs(matrix) == 'OK':
        numerical_outputs = await get_user_params(matrix, select, color, category)
    
        query = (f"Using the following user parameters {numerical_outputs}"
                "Generate summary reports for how the clothing will do in"
                "the market. Predict profit margin, quantity, and item total"
                "All kinds of predictions use clothing type, color, size"
                "Only quantity and item total use original price alongside"
                "the other predictors")
        
        response = await query_rag_system(query)
    
        return response
    else:
        return {'Sorry. Please Type in the Required Values'}
