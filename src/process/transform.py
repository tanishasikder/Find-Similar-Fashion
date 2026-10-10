import torch
from torchvision.transforms import v2

# Used the normalize the inputs
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class RandomGamma(torch.nn.Module):
    """random gamma transform"""
    def __init__(self, gamma_range=(0.7, 1.5), p=0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            gamma = torch.empty(1).uniform_(*self.gamma_range).item()
            img = v2.functional.adjust_gamma(img, gamma=gamma)
        return img

class RandomRGBShift(torch.nn.Module):
    """simulates per-channel color-temperature shift"""
    def __init__(self, shift_limit=15/255, p=0.5):
        super().__init__()
        self.shift_limit = shift_limit
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            shift = (torch.rand(3, 1, 1) * 2 - 1) * self.shift_limit
            img = (img + shift).clamp(0, 1)
        return img

def fashion_transform():
    '''
    Fashionpedia does not really train on color. Must transform
    images to test on different shades. Map with HEXCodes later.
    No hue jitter since the color label is made from the original hue
    and shifting it can turn a red label into an orange image.
    '''
    fashion_transforms = v2.Compose([
        v2.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # convert to [0,1] float BEFORE color ops
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        RandomGamma(gamma_range=(0.7, 1.5), p=0.5),
        RandomRGBShift(shift_limit=15/255, p=0.5),
        v2.RandomAutocontrast(p=0.2),        # closest native stand-in for CLAHE
        v2.GaussianNoise(mean=0.0, sigma=0.03),  # stand-in for ISONoise
        v2.Normalize(mean=mean, std=std),
    ])
    return fashion_transforms

def eval_transform():
    # No augmentation when testing so the scores are consistent
    return v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
