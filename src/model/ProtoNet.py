from torch import nn

class ProtoNet(nn.Module):
  def __init__(self, n_classes):
    super(ProtoNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1) # shape of output(16, H, W)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # shape of output(16, H/2, w/2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # shape of output(16, H/2, w/2)
    self.fc1 = nn.Linear(100352, 128)
    self.fc2 = nn.Linear(128, n_classes)
  # __init__():

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool(x)
    x = x.view(x.size(0), -1) # flattening
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    return x
  # forward
# ProtoNet