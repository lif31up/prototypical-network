import torch
from torch import nn
from config import Config

class ProtoNet(nn.Module):
  def __init__(self, config:Config):
    super(ProtoNet, self).__init__()
    self.config = config
    self.convs = self._create_convs(self.config.n_convs)
    self.act = nn.SiLU()
    self.flat = nn.Flatten()
    self.prototypes = None
  # __init__

  def _create_convs(self, n_convs):
    layers = nn.ModuleList()
    layers.append(
      nn.Conv2d(
        in_channels=self.config.input_channels,
        out_channels=self.config.hidden_channels,
        kernel_size=self.config.kernel_size,
        stride=self.config.stride,
        padding=self.config.padding)
    )  # first conv
    for i in range(n_convs - 2):
      layers.append(
        nn.Conv2d(
          in_channels=self.config.hidden_channels,
          out_channels=self.config.hidden_channels,
          kernel_size=self.config.kernel_size,
          padding=self.config.padding,
          stride=self.config.stride,
          bias=self.config.bias),
      )  # hidden convs
    layers.append(
      nn.Conv2d(
        in_channels=self.config.hidden_channels,
        out_channels=self.config.output_channels,
        kernel_size=self.config.kernel_size,
        stride=self.config.stride,
        padding=self.config.padding)
    )  # last conv
    return layers
  # _create_convs

  def get_prototypes(self, support_set):
    self.prototypes = get_prototypes(support_set)
    return self.prototypes
  # get_prototypes

  def forward(self, x):
    assert self.prototypes is not None, "self.prototypes is None"
    x = self.convs[0](x)
    x = self.act(x)
    for conv in self.convs[1:-1]:
      res = x
      x = conv(x)
      x = self.act(x)
      x += res
    x = self.convs[-1](x)
    return self.cdist(x, self.prototypes)
  # forward

  def cdist(self, x, prototypes):
    flatten_x = self.flat(x)
    return torch.cdist(flatten_x, prototypes, p=2)
  # cdist
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

class PRNLoss(nn.Module):
  def __init__(self, config:Config, reduction='mean'):
    super(PRNLoss, self).__init__()
    self.config = config
    self.reduction = reduction
    self.prototypes = None
  # __init__

  def set_prototypes(self, prototypes): self.prototypes = prototypes

  def forward(self, pred_y, act_y):
    assert self.prototypes is not None, "self.prototypes is None"
    nom = torch.exp(-1 * pred_y[0][torch.argmax(act_y, dim=1)])
    denom = torch.sum(torch.exp(torch.negative(pred_y)), dim=1, keepdim=True)
    loss = torch.div(nom, denom)
    return loss.mean() if self.reduction == "mean" else loss.sum() if self.reduction == "sum"
  # forward
# PRNLoss