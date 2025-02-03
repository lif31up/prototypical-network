import random
import torch

import torchvision as tv
from torch.utils.data import Dataset, DataLoader, TensorDataset


def euclidean_distance(a: torch.tensor, b: torch.tensor):
  if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): raise ValueError(
    "a and b must be a PyTorch tensor.")
  n, m = a.shape[0], b.shape[0]
  a, b = a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)
  return torch.pow(a - b, 2).sum(2)


# euclidean_distance()

class FSLDataset(Dataset):
  def __init__(self, dataset, mode="support"):
    self.dataset = [] if not dataset else dataset
    self.mode = mode

  def __getitem__(self, idx):
    feature, label = self.dataset[idx]
    return feature, label
  # __getitem__():

  def __len__(self): return len(self.dataset)
# FSLDataset()

class Episoder:
  def __init__(self, dataset, n_way, k_shot, n_query, transform):
    self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query  # define n-way/k-hot framework parameters
    self.dataset, self.transform = dataset, transform  # init dataset and apply transformer
    self.n_classes, self.ways = self.init()
  # __init__()

  def init(self):
    if len(self.dataset) < 1: raise ValueError("dataset has invalid value.")
    n_classes = len(self.dataset.classes)
    indices_of_classes = [[] for _ in range(n_classes)]
    for index, (feature, label) in enumerate(self.dataset):
      try: indices_of_classes[label].append(index)
      except ValueError as e: raise ValueError(f"{e}:indices_of_classes has invalid value.")
    # for
    return n_classes, indices_of_classes
  # analyze

  def get_episode(self):  # select classes using list of chosen indexes
    support_set, query_set = [], []
    # build support_set
    for way in range(0, self.n_classes):
      for cnt, index in enumerate(random.sample(self.ways[way], self.k_shot)):
        support_set.append(self.dataset[index])
        self.ways[way].pop(cnt)
    # build query_set
    query_set = support_set.copy()
    for way in range(0, self.n_classes):
      for cnt, index in enumerate(random.sample(self.ways[way], self.n_query)):
        query_set.append(self.dataset[index])
        self.ways[way].pop(cnt)
    # for for
    return FSLDataset(support_set, mode="support"), FSLDataset(query_set, mode="query")
  # get_episode()
# Episoder()

def main(path: str):
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = Episoder(imageset, 3, 4, 4, transform)
  support_set, query_set = episoder.get_episode()
# main():

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean")
