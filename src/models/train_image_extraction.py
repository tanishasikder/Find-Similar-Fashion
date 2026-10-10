import mlflow.pytorch
import torch
import os
import copy
import json
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import mlflow
from pathlib import Path
import sys
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1]

sys.path.insert(0, str(root_dir)) # Fix the path before importing below classes

from src.core.tracking_config import Dagshub_Track
from src.models.image_extraction import CNN, device
from src.models.image_data import ImageData, get_class_names
from src.process.transform import fashion_transform, eval_transform

load_dotenv()

image_dir = os.environ.get('CROPPED_IMAGES')
image_path = os.environ.get('IMAGE_MODEL')

sets = ['train', 'test']

# Train on the heads of the vgg16 to learn clothing features
def train_model(model, criterion, optimizer, scheduler, num_epochs=None):
    color_criterion, cat_criterion, attr_criterion = criterion
    with mlflow.start_run():
        best_model = copy.deepcopy(model.state_dict())
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            # Switch between training and validation
            for phase in sets:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                run_loss = 0.0
                color_correct = 0
                cat_correct = 0
                attr_correct = 0.0

                # Loop over the labels and the images in the dataloader
                for input, color, cat, attr in tqdm(fashion_loaders[phase], desc=phase):
                    input = input.to(device, non_blocking=True)
                    color, cat, attr = color.to(device), cat.to(device), attr.to(device)

                    with torch.set_grad_enabled(phase=='train'):
                        # Gets the outputs from vgg16 model
                        color_pred, cat_pred, attr_pred = model(input)
                        # Crossentropy loss expects raw scores
                        color_loss = color_criterion(color_pred, color)
                        cat_loss = cat_criterion(cat_pred, cat)
                        attr_loss = attr_criterion(attr_pred, attr)

                        # Overall loss from all predictions
                        loss = color_loss + cat_loss + attr_loss

                        # Optimizes and backward propagates if it is training
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # Gets the largest score for single label accuracy
                    color_correct += (color_pred.argmax(1) == color).sum().item()
                    cat_correct += (cat_pred.argmax(1) == cat).sum().item()
                    # Attributes are multi label so each one is on if its probability > 0.5
                    attr_hits = ((torch.sigmoid(attr_pred) > 0.5) == attr.bool()).float()
                    attr_correct += attr_hits.mean(1).sum().item()

                    # Calculates the loss and correct labels
                    run_loss += loss.item() * input.size(0)

                # Overall loss and accuracy of this model
                size = dataset_sizes[phase]
                epoch_loss = run_loss / size
                color_acc = color_correct / size
                cat_acc = cat_correct / size
                attr_acc = attr_correct / size
                epoch_accuracy = (color_acc + cat_acc + attr_acc) / 3

                print(f'{phase} Loss : {epoch_loss:.3f} Accuracy : {epoch_accuracy:.3f} '
                      f'(color {color_acc:.3f}, category {cat_acc:.3f}, attributes {attr_acc:.3f})')
                mlflow.log_metrics({
                    f'{phase}_loss': epoch_loss,
                    f'{phase}_color_accuracy': color_acc,
                    f'{phase}_category_accuracy': cat_acc,
                    f'{phase}_attribute_accuracy': attr_acc,
                }, step=epoch)

                # If it is validation, find the best model by finding the best accuracy
                if phase == 'test' and epoch_accuracy > best_accuracy:
                    best_accuracy = epoch_accuracy
                    best_model = copy.deepcopy(model.state_dict())
                    mlflow.log_metric('best accuracy', best_accuracy)
                    model_info = mlflow.pytorch.log_model(model, name='europe-fashion-image-extract')
                    mlflow.register_model(model_uri=f"models:/{model_info.model_id}", name='europe-fashion-image-extract')

            scheduler.step()
        print(f'Best model accuracy: {best_accuracy:.3f}')

        # save the best model's weights and the class names needed to decode its outputs
        save_dir = Path(image_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_model, save_dir / 'image_extraction_model.pth')
        co_names, cat_names, attr_names = class_names
        with open(save_dir / 'image_extraction_classes.json', 'w', encoding='utf-8') as f:
            json.dump({'color': co_names, 'category': cat_names, 'attribute': attr_names}, f)

# Required for multi-processing in Windows
# Executes code that start everything
if __name__ == '__main__':
    print(f'Training on {device}')
    # Use the DagsHub Mlflow server to log things
    track = Dagshub_Track()
    track.initialize()

    # Finding all the images in the folder. Only training images get augmented
    train_data = ImageData(transform=fashion_transform())
    test_data = ImageData(transform=eval_transform())

    # Splitting the dataset into train test by source photo, not by crop. Crops of the
    # same photo overlap (a shirt and its collar) so splitting per crop leaks test into train
    sources = train_data.source_ids()
    unique_sources = sorted(set(sources))
    order = torch.randperm(len(unique_sources), generator=torch.Generator().manual_seed(42)).tolist()
    test_sources = {unique_sources[i] for i in order[int(0.8 * len(unique_sources)):]}

    train_idx = [i for i, s in enumerate(sources) if s not in test_sources]
    test_idx = [i for i, s in enumerate(sources) if s in test_sources]

    train = train_data.subset(train_idx)
    test = test_data.subset(test_idx)
    train_size, test_size = len(train), len(test)

    # Loading the data in batches
    pin = device.type == 'cuda'
    # Enough workers to keep the gpu busy. persistent so Windows does not respawn them every epoch
    loader_args = dict(batch_size=32, num_workers=8, pin_memory=pin, persistent_workers=True)
    fashion_loaders = {
        'train' : DataLoader(train, shuffle=True, **loader_args),
        'test' : DataLoader(test, shuffle=False, **loader_args)
    }

    dataset_sizes = {'train': train_size, 'test': test_size}

    # The heads have to match how the labels were encoded in the dataset
    class_names = get_class_names()
    model = CNN(*class_names)

    # Color and category are single label so crossentropy
    # Attribute is multi label so bce loss
    criterion = (nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss())
    # Only the heads are trained since the vgg16 features are frozen
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)

    # Every 7 epochs the learning rate is multiplied by gamma
    step_lr = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initializing the final model with all the parameters
    train_model(model, criterion, optimizer, step_lr, num_epochs=5)
