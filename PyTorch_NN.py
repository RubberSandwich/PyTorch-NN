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
import torch.optim as optim
import datetime, time

#Monochrome, 28x28 bitmap, pen tool 3px thickness, a-z lower case, 26 letters abcdef ghijklmnop qrstuvxyz

#Coarse parameters
train=False
epochs=1

device = 'cuda' if torch.cuda.is_available() else 'cpu' #TODO: Something with assertion while using cuda breaks, compile with TORCH_USE_CUDA_DSA suggestion

data_path = Path("data/letter_classification/")
train_dir = data_path / "train"
test_dir = data_path / "test"

class LetterNet(nn.Module):
    def __init__(self):
        super(LetterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 28)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(50,60,5)
        self.fc1 = nn.Linear(60*5*5,200)
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = LetterNet()
net.to(device)

data_transform = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()
    ])

training_data = datasets.EMNIST(root='./data/letter_classification/',
                                            split='letters',
                                            train=True, 
                                            download=True, 
                                            transform=data_transform)

testing_data = datasets.EMNIST(root='./data/letter_classification/',
                                            split='letters',
                                            train=False, 
                                            download=True, 
                                            transform=data_transform)

training_dataloader = DataLoader(dataset=training_data,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

testing_dataloader = DataLoader(dataset=testing_data,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0)

letter_classes = ('a','b','c','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')

learning_rate = 1e-3

inputs, labels = training_data[0], training_data[1]

print(f'Input: {inputs}, Label: {labels}')

if train:
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(net.parameters(),lr=learning_rate,momentum=.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(training_dataloader,0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimiser.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if i% 2500 == 2499: #Print every 2500 mini batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2500:.3f}')
                running_loss = 0.0
    print('Finished training!')

    save_path = "./savedstates/letter_net.pth"

    torch.save(net.state_dict(), save_path)
    print(f'Saved state as {save_path}')