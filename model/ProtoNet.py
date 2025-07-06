import torch
from torch import nn

class ProtoNet(nn.Module):
  def __init__(self, config):
    super(ProtoNet, self).__init__()
    self.config, self.prototypes = config, None
    self.flatten, self.act, self.softmax = nn.Flatten(1), nn.SiLU(), nn.Softmax(dim=1)

    self.conv1 = nn.Conv2d(in_channels=config["in_channels"], out_channels=config["hidden_channels"], kernel_size=config["kernel_size"], stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=config["hidden_channels"], out_channels=config["hidden_channels"], kernel_size=config["kernel_size"], stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=config["hidden_channels"], out_channels=config["out_channels"], kernel_size=config["kernel_size"], stride=1, padding=1)
  # __init__():

  def forward(self, x):
    assert self.prototypes is not None, "self.prototypes is None"
    x = self.conv1(x)
    x = self.conv2(self.act(x))
    x = self.conv3(self.act(x))
    x = self.cdist(x, self.prototypes)
    return self.softmax(x)
  # forward():

  def cdist(self, x, prototypes):
    flatten_x = self.flatten(x)
    return torch.cdist(flatten_x, prototypes, p=2)
  # cdist():
# ProtoNet

def get_prototypes(support_set):
  prototypes = list()
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set:
    idx = support_set.classes.index(label)
    embedded_features_list[idx].append(embedded_feature)
  for embedded_features in embedded_features_list:
    class_prototype = torch.stack(embedded_features).mean(dim=0)
    prototypes.append(class_prototype.flatten())
  prototypes = torch.stack(prototypes)
  return prototypes
# get_prototypes