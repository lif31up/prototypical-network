import random

from torch.utils.data import Dataset

class FewShotDataset(Dataset):
  def __init__(self, dataset, n_way, n_shot, n_query, transform=None, mode="support"):
    self.dataset = dataset
    self.n_way, self.n_shot, self.n_query = n_way, n_shot, n_query
    self.transform = transform
    self.mode = mode
  # __init__()

  def __len__(self): return len(self.dataset)

  def __getitem__(self, index):
    feature, label = self.dataset[index]
    if self.mode == "query": label = -1
    return feature, label
  # __getitem__()
# FewShotDataset()