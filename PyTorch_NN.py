import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1,6,5)
		self.conv2 = nn.Conv2d(6,16,5)

		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):
		
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()

learning_rate = 1e-2
optimizer = optim.SGD(net.parameters(),learning_rate)
optimizer.zero_grad()

input = torch.rand(1,1,32,32)
out = net(input)

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(out, target)

loss.backward()

print('conv1.weight before optim')
print(net.conv1.weight)

optimizer.step()

print('conv1.weight before optim')
print(net.conv1.weight)