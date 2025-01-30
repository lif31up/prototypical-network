from torch import nn

class ProtoNet(nn.Module):
  def __init__(self, inpt, hidn, oupt):
    super(ProtoNet, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(inpt, hidn, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(hidn, hidn, 3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(hidn * 5 * 5, oupt)
    )
  # __init__()

  def forward(self, x): return self.encoder(x)
# ProtoNet