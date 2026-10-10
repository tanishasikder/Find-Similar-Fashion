from PIL import Image
from pathlib import Path
import os
import json
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import numpy as np
from dotenv import load_dotenv
from torch.utils.data import Dataset
import torch

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2] # Project root, where .env lives

files = BASE_DIR / os.getenv('CROPPED_CSV')

# Load in images in the order of the csv file
class FashionData(Dataset):
    def __init__(self):
        xyz = np.loadtxt(files, delimiter=',', dtype=np.float32)
        self.images = torch.from_numpy(xyz[:, 0]) # First column has file names
        self.cat = torch.from_numpy(xyz[:, [1]]) # Second label has categories
        self.attr = torch.from_numpy(xyz[:, [2]]) # Third has attributes
        # Double bracket the labels to make it 2d
        self.n_samples = xyz.shape[0] # Stores number of data points

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        # Accesses a single sample by index
        return self.images[index], self.cat[index], self.attr[index]
