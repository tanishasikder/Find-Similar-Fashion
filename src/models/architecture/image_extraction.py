import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

# Push to GPU if it is available, CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN class to classify image features
class CNN(nn.Module):
    def __init__(self, co_names, cat_names, attr_names):
        super().__init__()
        self.color_names = co_names
        self.category_names = cat_names
        self.apparels = attr_names
        # Load in the pretrained resnet model
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)

        # Freeze parameters
        for param in model.features.parameters():
            param.requires_grad = False

        self.vgg16_features = model.features
        self.avgpool = model.avgpool
        # Assign a fully connected layer containing the class names
        num_features = 512 * 7 * 7
        # Head to classify the color
        self.fc_color = nn.Linear(num_features, len(co_names))
        self.dropout1 = nn.Dropout(0.5)
        # Head to classify the clothing category
        self.fc_category = nn.Linear(num_features, len(cat_names))
        self.dropout2 = nn.Dropout(0.5)
        self.fc_attr = nn.Linear(num_features, len(attr_names))
        self.dropout3 = nn.Dropout(0.5)
        self.to(device)
    
    def forward(self, x):
        # Gather features and assign it to the heads
        x = self.vgg16_features(x)
        x = self.avgpool(x)
        # Flatten the features so it can be used in linear layers
        # Goes from [batch, 512, 1, 1] to [batch, 512]
        x = torch.flatten(x, 1)
        color = self.fc_color(self.dropout1(x))
        category = self.fc_category(self.dropout2(x))
        attr = self.fc_attr(self.dropout3(x))
        # Return the classification

        return color, category, attr