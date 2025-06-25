import torch
from torch import nn

def stack(in_channels, out_channels, kernel_size):
  return nn.Sequential(
    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
    nn.SiLU(),
  )  # nn.Sequential
# stack():

class ProtoNet(nn.Module):
  def __init__(self, config):
    super(ProtoNet, self).__init__()
    self.config = config
    self.flatten = nn.Flatten()
    self.stacks = nn.Sequential(
      stack(config["in_channels"], config["hidden_channels"], config["kernel_size"]),
      #stack(config["hidden_channels"], config["hidden_channels"], config["kernel_size"]),
      #stack(config["hidden_channels"], config["hidden_channels"], config["kernel_size"]),
      stack(config["hidden_channels"], config["out_channels"], config["kernel_size"]),
    )
  # __init__():

  def cdist(self, x, prototypes):
    flatten_x = self.flatten(x).unsqueeze(1)
    flatten_prototypes = prototypes.reshape(prototypes.shape[0], -1).unsqueeze(0)
    distances = torch.norm(flatten_prototypes - flatten_x, p=2, dim=-1)
    return distances
  # cdist():

  def forward(self, x, prototypes):
    x = self.stacks(x)
    return self.cdist(x, prototypes)
  # forward():
# ProtoNet

def get_prototypes(support_set, seen_classes):
  prototypes = []
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set: embedded_features_list[seen_classes.index(label)].append(embedded_feature)
  for embedded_features in embedded_features_list:
    class_prototype = torch.stack(embedded_features).mean(dim=0)
    prototypes.append(class_prototype)
  return torch.stack(prototypes)
# get_prototypes