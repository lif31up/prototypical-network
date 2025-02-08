import torch
from torch import nn
import torchvision as tv
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from tqdm import tqdm

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

# mian function to simulate the few-shot learning pipeline.
def main(path: str):
  # define the image transformations
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]) # transform

  # load the dataset
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, 2, 2, transform)

  # init ProtoNet
  model = ProtoNet()

  # simulate training flow
  epochs, iters = 2, 2
  for _ in range(epochs):
    support_set, query_set = episoder.get_episode()
    prototypes = list()
    for _ in tqdm(range(iters)):
      # compute prototype from support examples
      embedded_features_list = [[] for _ in range(len(support_set.classes))]
      for embedded_feature, label in support_set: embedded_features_list[label].append(embedded_feature)
      for embedded_features in embedded_features_list:
        sum = torch.zeros_like(embedded_features[0])
        for embedded_feature in embedded_features: sum += embedded_feature
        sum /= len(embedded_features)
        prototypes.append(sum.requires_grad_(True))
      # update loss
      model.prototyping(prototypes)
      for feature, label in DataLoader(query_set, shuffle=True):
        print(model.forward(feature), label)
  # for for
# main():

if __name__ == "__main__": main("../../data/raw/omniglot-py/images_background/Korean")