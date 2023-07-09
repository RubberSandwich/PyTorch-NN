import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Monochrome, 32x32 bitmap, pen tool 3px thickness, a-z lower case, 26 letters abcdef ghijklmnop qrstuvxyz

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = Path("data/letter_classification/")
train_dir = data_path / "train"
test_dir = data_path / "test"

class LetterNet(nn.Module):
    def __init__(self):
        super(LetterNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = LetterNet()

random.seed(32)

data_path_list = list(data_path.glob("*/*/*.jpg"))

random_image_path = random.choice(data_path_list)

image_class = random_image_path.parent.stem

data_transform = transforms.Compose([
    #transforms.TrivialAugmentWide(),
    transforms.ToTensor()
    ])

training_data = datasets.ImageFolder(root=train_dir,
                                          transform=data_transform,
                                          target_transform=None)

#testing_data = datasets.ImageFolder(root=test_dir,
#                                    transform=data_transform)
#Not made yet :)

training_dataloader = DataLoader(dataset=training_data,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=True)

img, label = next(iter(training_dataloader))

print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")