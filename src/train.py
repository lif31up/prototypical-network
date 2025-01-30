import random

import torch.cuda
import torchvision as tv
from torch import nn
from torch.utils.data import Dataset, DataLoader
from data.FewShotDataset import FewShotDataset
from model.ProtoNet import ProtoNet

def main(path, save_to, epochs=10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform

  imageset = tv.datasets.ImageFolder(root="./data/raw/omniglot-py/images_background/Korean")

  # divide imageset into support_set and query_set
  support_set = FewShotDataset(imageset, 5, 5, 15, transform, mode="support_set")
  ## trainset = DataLoader(, batch_size=32, shuffle=True)

  """
  model = ProtoNet(1, 64, 64).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(epochs):
    for x, y in trainset:
      loss = criterion(model(x), y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  # for for
  """
# main()

if __name__ == "__main__": main("./data/raw/", "")