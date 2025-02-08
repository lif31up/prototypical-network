import torch
from torch import nn

class ProtoNet(nn.Module):
  def __init__(self):
    super(ProtoNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.softmax = nn.LogSoftmax(dim=0)
  # __init__():

  def prototyping(self, prototypes): self.prototypes = prototypes

  def cdist(self, x):
    dists = torch.cdist(x, self.prototypes, p=2).squeeze(0)  # Efficient batch-wise L2 distance computation
    return dists
  # cdist()

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.flatten(x)
    x = self.cdist(x)
    x = self.softmax(x)
    return x
  # forward
# ProtoNet