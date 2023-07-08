import torch
import torch.nn as nn
import torch.nn.functional as F

#Monochrome, 32x32 bitmap, pen tool 3px thickness, a-z lower case, 26 letters abcdef ghijklmnop qrstuvxyz

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LetterNet(nn.Module):
    def __init__(self):
        super(LetterNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)

    def forward(self, x):
        x = self.conv1(x)
        return x

net = LetterNet()

print(net)