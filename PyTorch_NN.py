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

#Monochrome, 32x32 bitmap, pen tool 3px thickness, a-z lower case, 26 letters abcdef ghijklmnop qrstuvxyz

#Coarse parameters
train=False #False means testing mode
save=False
epochs=50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = Path("data/letter_classification/")
train_dir = data_path / "train"
test_dir = data_path / "test"
save_path = "./savedstates/letter_net.pth"

class LetterNet(nn.Module):
    def __init__(self):
        super(LetterNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 25, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(25,35,5)
        self.fc1 = nn.Linear(35*5*5,120)
        self.fc2 = nn.Linear(120, 84)
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

letter_classes = ('a','b','c','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z')

learning_rate = 1e-3

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
            if i% 250 == 249: #Print every 250 mini batches
                print(f"Data label: {labels[0]}")
                print(f"Model prediction: {outputs[0]}")

                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 250:.3f}')
                running_loss = 0.0
    print('Finished training!')

    if save:
        torch.save(net.state_dict(), save_path)
        print(f'Saved state as {save_path}')
    print('Exiting')

else:
    net.load_state_dict(torch.load(save_path))
    net.eval()

    loss_fn = nn.CrossEntropyLoss()

    test_single = datasets.ImageFolder(root=test_dir / "single", 
                                       transform=data_transform)
    
    test_single_load = DataLoader(dataset=test_single, batch_size=1, 
                                  num_workers=0, shuffle=True)

    with torch.no_grad():
        for i,data in enumerate(test_single_load,0):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)

            print(outputs.type(torch.float).sum().item())