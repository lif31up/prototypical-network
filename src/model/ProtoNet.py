import torch
from torch import nn

class ProtoNet(nn.Module):
  def __init__(self, config):
    super(ProtoNet, self).__init__()
    self.config = config
    self.flatten, self.act, self.softmax = nn.Flatten(1), nn.SiLU(), nn.LogSoftmax(dim=1)

    self.conv1 = nn.Conv2d(in_channels=config["in_channels"], out_channels=config["hidden_channels"], kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=config["hidden_channels"], out_channels=config["hidden_channels"], kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=config["hidden_channels"], out_channels=config["out_channels"], kernel_size=3, stride=1, padding=1)
  # __init__():

  def forward(self, x, prototypes):
    x = self.act(self.conv1(x))
    x = self.act(self.conv2(x))
    x = self.conv3(x)
    x = self.cdist(x, prototypes)
    return self.softmax(-1 * x)
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