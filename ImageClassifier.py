import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Train = False
epochs = 5

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 24, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(24, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()

net.to(device)

def imshow(img):
    imgt = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
										download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
											shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
			'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

PATH = './cifar_net.pth'

if Train:
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	for epoch in range(epochs):

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data[0].to(device), data[1].to(device)
	
			optimizer.zero_grad()
	
			outputs = net(inputs)
	
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			if i% 2000 == 1999: #Print every 2000 mini batches
				print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
				running_loss = 0.0

	print('Finished training')

	torch.save(net.state_dict(), PATH)
	print(f'Saved state as {PATH}')

else:
	dataiter = iter(testloader)
	images, labels = next(dataiter)

	net.load_state_dict(torch.load(PATH))

	correct_pred = {classname: 0 for classname in classes}
	total_pred = {classname: 0 for classname in classes}

	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predictions = torch.max(outputs, 1)
		
			for label, prediction in zip(labels,predictions):
				if label == prediction:
					correct_pred[classes[label]] += 1
				total_pred[classes[label]] += 1

	for classname, correct_count in correct_pred.items():
		accuracy = 100 * float(correct_count) / total_pred[classname]
		print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')