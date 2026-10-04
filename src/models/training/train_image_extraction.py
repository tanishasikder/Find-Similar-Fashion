import mlflow.pytorch
from sentence_transformers import SentenceTransformer
import torch
import os
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import mlflow
import dagshub
import numpy as np
from PIL import Image
from pathlib import Path
import sys
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

'''
Seems like all of the file names are in the data but for some reason it is not
finding the file names. Debug to see where it is failing to find the file names
'''
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[2]

sys.path.insert(0, str(root_dir)) # Fix the path before importing below classes

from src.process.process_color import get_color_data, get_colors
from src.core.tracking_config import Dagshub_Track
from src.models.architecture.image_extraction import CNN
from src.process.process_color import ColorData, get_color_data
from src.process.process_image import get_type_labels
from src.models.data_models.image_data import ImageData

load_dotenv()

image_dir = os.environ.get('CROPPED_IMAGES')
annotation = os.environ.get('ANNOTATION_DIR') # Might not need this we'll see
image_path = os.environ.get('IMAGE_MODEL')

train_labels = os.environ.get('TRAIN_LABELS')
sets = ['train', 'test']

# Train on the last layer of the resnet to learn clothing features
def train_model(model, criterion, optimizer, scheduler, num_epochs=None):
    with mlflow.start_run():
        best_model = model.state_dict()
        best_accuracy = 0.0
        print('we training now')
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            # Switch between training and validation
            for phase in sets:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                run_loss = 0.0
                loss = 0.0
                correct = 0

                # Loop over the labels and the images in the dataloader
                for input, cat, attr in fashion_loaders[phase]:
                    with torch.set_grad_enabled(phase=='train'):
                        # Gets the outputs from resnet model
                        print(len(input[0]))
                        color_pred, cat_pred, attr_pred = model(input)
                        print('we modeling now')
                        # Crossentropy loss expects raw scores
                        # COLOR NEEDS TO COME FROM ANOTHER SOURCE ITS NOT LABEL[:, 0]
                        #LABEL[:, 0] IS THE FILENAME
                        #color_loss = criterion(color, label[:, 0])
                        # Match the shape of the label
                        new_cat = cat.float().unsqueeze(1)
                        new_attr = attr.float().unsqueeze(1)

                        cat_loss = criterion(cat_pred, new_cat)
                        new_attr = attr.float().unsqueeze(1)
                        attr_loss = criterion(attr_pred, )

                        # Gets the largest score for accuracy
                        #_, color_pred = torch.max(color_pred, 1)
                        _, cat_pred = torch.max(cat_pred, 1)
                        _, attr_pred = torch.max(attr_pred, 1)

                        # Overall loss from both predictions
                        loss = cat_loss #+ color_loss + attr_loss

                        # Optimizes and backward propagates if it is training
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    
                    # Calculates the loss and correct labels
                    run_loss += loss.item() * input.size(0)
                    #correct += torch.sum(color_pred == (label[:, 0]))
                    correct += torch.sum(cat_pred == cat)
                    correct += torch.sum(attr_pred == attr)
            
                # Overall loss and accuracy of this model
                epoch_loss = run_loss / dataset_sizes[phase]
                #print(f'data_size phase{dataset_sizes}')
                epoch_accuracy = correct / (3 * dataset_sizes[phase])

                print(f'{phase} Loss : {epoch_loss:.3f} Accuracy : {epoch_accuracy:.3f}')

                # If it is validation, find the best model by finding the best accuracy
                if phase == 'test' and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    mlflow.log_metric('best accuracy', best_accuracy)
                    model_info = mlflow.pytorch.log_model(model, name='europe-fashion-image-extract')
                    mlflow.register_model(model_uri=f"models:/{model_info.model_id}", name='europe-fashion-image-extract')
                    best_model = mlflow.pytorch.load_model(model_uri=f"models:/{model_info.model_id}")

            scheduler.step()
        print(f'Best model accuracy: {best_accuracy:.3f}')
        torch.save(best_model.state_dict(), image_path)   # save the best model's weights and params

# Required for multi-processing in Windows
# Executes code that start everything
if __name__ == '__main__':
    # Use the DagsHub Mlflow server to log things
    track = Dagshub_Track() 
    track.initialize()

    # Finding all the images in the folder
    dataset = ImageData()

    print('got dataset')
    # Splitting the dataset into train test
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    #train, test = random_split(dataset, [train_size, test_size])
    #train, test = train_test_split(dataset, train_size=0.8, test_size=0.2, shuffle=False)
    train = Subset(dataset, range(0, train_size))
    test = Subset(dataset, range(train_size, len(dataset)))
    #train = dataset[:train_size]
    #test = dataset[train_size:]
    # Loading the data in batches. Separate dataloaders for color and type tests
    fashion_loaders = {
        'train' : DataLoader(train, batch_size=32, shuffle=False, num_workers=4, pin_memory=True),
        'test' : DataLoader(test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)                       
    }  
    '''
    for idx, (inputs, labels) in fashion_loaders['train']:
        print(f"--- Batch {idx} ---")
        print("Inputs:", inputs)
        print("Labels:", labels)   
    '''
    dataset_sizes = {'train': train_size, 'test': test_size}

    # Configuring with color and clothing classes. Removing dashes
    codes = get_colors() # Dict where the values have the colors as strings
    clothing, attr = get_type_labels()

    model = CNN(list(codes.values()), clothing, attr)
    print('i guess model worked lmao')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Every 7 epochs the learning rate is multiplied by gamma
    step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
    # Initializing the final model with all the parameters
    train_model(model, criterion, optimizer, step_lr, num_epochs=10)