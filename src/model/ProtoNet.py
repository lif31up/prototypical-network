import torch
from torch import nn
import torchvision as tv
from src.FewShotEpisoder import FewShotEpisoder
from src.utils import softmax_dist


class ProtoNet(nn.Module):
  def __init__(self):
    super(ProtoNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
  # __init__():

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.conv4(x)
    return x
  # forward
# ProtoNet

def main(path: str):
  # define the image transformations
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])  # transform

  # load the dataset
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, 2, 2, transform)
  support_set, query_set = episoder.get_episode()

  model = ProtoNet()
  feature, label = support_set[0]
  result = model.forward(feature, support_set.prototypes)
  print(result.size(), support_set.prototypes[0].size())
# main()

if __name__ == "__main__": main("../../data/raw/omniglot-py/images_background/Korean")