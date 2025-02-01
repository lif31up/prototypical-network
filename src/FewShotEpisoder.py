import random
import torch

import torchvision as tv
from torch.utils.data import Dataset, DataLoader, TensorDataset


def euclidean_distance(a: torch.tensor, b: torch.tensor):
  if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): raise ValueError("a and b must be a PyTorch tensor.")
  n, m = a.shape[0], b.shape[0]
  a, b = a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)
  return torch.pow(a - b, 2).sum(2)
# euclidean_distance()

def avg_embedding(embeddings: torch.tensor):
  if not isinstance(embeddings, torch.Tensor): raise ValueError("class_embedding must be a PyTorch tensor.")
  return embeddings.mean(dim=0)
# avg_embedding(x)

class FSLDataset(Dataset):
  def __init__(self, dataset, mode="support"):
    self.dataset = [] if not dataset else dataset
    self.mode = mode
  def __getitem__(self, idx):
    feature, label = self.dataset[idx]
    if self.mode == "query": label = -1
    return feature, label

  def add_item(self, item): self.dataset.append(item)

  def __len__(self): return len(self.dataset)

class Episoder:
  def __init__(self, dataset, n_way, k_shot, n_query, transform):
    self.dataset = dataset
    self.transform = transform
    self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
    self.n_classes = len(dataset.classes)
    self.ways = self.get_ways()
    self.prototypes = self.get_prototypes()
  # __init__()

  def get_ways(self):
    ways = [FSLDataset([]) for _ in range(self.n_classes)]
    for idx, item in enumerate(self.dataset):
      feature, label = item
      ways[label].add_item(item)
    # for
    return ways
  # get_ways()

  def get_prototypes(self):
    list_of_prototypes = []
    for way in self.ways:
      sum = 0
      for feature, _ in way:
        x = self.transform(feature)
        sum += x
      # for
      prototype = sum / len(way)
      list_of_prototypes.append(prototype)
    # for
    return list_of_prototypes
  # get_prototypes():

  def get_episode(self, selected_ways: list, selected_query: list):
    if not self.dataset: raise TypeError("dataset is not provided.")
    support_set, query_set = FSLDataset([], mode="support"), FSLDataset([], mode="query")
    for way in selected_ways:
      for idx, (feature, label) in enumerate(self.ways[way]):
        if idx >= self.k_shot: break
        support_set.add_item((feature, self.prototypes[label]))
    # for # for
    for way in selected_query:
      for idx, (feature, label) in enumerate(self.ways[way]):
        if idx >= self.n_query: break
        query_set.add_item((feature, self.prototypes[label]))
    # for for

    return support_set, query_set
  # get_episode()
# Episoder()

def main(path: str):
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])  # transform
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = Episoder(imageset, 3, 4, 4, transform)

  epochs, iters = 4, 2
  for _ in range(epochs):
    support, query = episoder.get_episode([3, 4, 5], [3, 4, 6])
    for _ in range(iters):
      for x, y in DataLoader(support, shuffle=True):
        print(x.size())
    # for # for
    print("---")
    for _ in range(iters):
      for x, y in DataLoader(query, shuffle=True):
        print(x.size())
    # for # for
# main():

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean")