import torch
from torch import nn
import torch.nn.functional as torch_f

class ProtoNet(nn.Module):
  def __init__(self, in_channels=3, hidden_channel=26, output_channel=3):
    super(ProtoNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels, hidden_channel, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(hidden_channel, output_channel, kernel_size=3, stride=1, padding=1)
    self.swish = nn.SiLU()
    self.flatten = nn.Flatten()
    self.softmax = nn.LogSoftmax(dim=1)
  # __init__():

  def prototyping(self, prototypes): self.prototypes = prototypes

  def cdist(self, x: torch.Tensor, metric="euclidean") -> torch.Tensor:
    assert self.prototypes is not None, "Prototypes must be set before calling cdist."
    assert x.size(1) == self.prototypes.size(1), "Feature dimensions must match."
    if metric == "euclidean":
      dists = torch.cdist(x, self.prototypes, p=2)  # L2 distance
    elif metric == "cosine":
      dists = 1 - torch_f.cosine_similarity(x.unsqueeze(1), self.prototypes.unsqueeze(0), dim=2)  # 1 - cosine similarity
    else:
      raise ValueError("Unsupported distance metric. Choose 'euclidean' or 'cosine'.")
    return dists
  # cdist()

  def forward(self, x):
    x = self.conv1(x)
    x = self.swish(x)
    x = self.conv2(x)
    x = self.swish(x)
    x = self.conv3(x)
    x = self.swish(x)
    x = self.flatten(x)
    x = self.cdist(x, metric="euclidean")
    return self.softmax(-x)
  # forward
# ProtoNet