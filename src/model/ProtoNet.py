import torch
from torch import nn
import torchvision as tv
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from tqdm import tqdm


class ProtoNet(nn.Module):
  def __init__(self):
    super(ProtoNet, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64, 224, 224)
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # (128, 224, 224)
      nn.ReLU(),
    ) # encoder (down-scaling)
    self.decoder = nn.Sequential(
      nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (64, 224, 224)
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),  # (3, 224, 224)
    ) # decoder (up-scaling)
  # __init__():

  def forward(self, x):
    self.encoder(x)
    self.decoder(x)
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
      for feature, label in DataLoader(query_set.prototyping(prototypes), shuffle=True):
        print(model.forward(feature).size(), label.size())
  # for for
# main():

if __name__ == "__main__": main("../../data/raw/omniglot-py/images_background/Korean")