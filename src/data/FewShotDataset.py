import random

import torch
from torch.utils.data import Dataset

class FewShotEpisoder:
  def __init__(self, dataset, n_way, k_shot, n_query, transform=None):
    self.dataset, self.transform = dataset, transform
    self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
    self.ways = self.get_ways(self.dataset.class_to_idx.values())
  # __init__():

  def get_ways(self, class_to_idx):
    list_of_labeled_indices = list()
    for cls in class_to_idx:
      list_of_labeled_indices.append([idx for idx, (item, label) in enumerate(self.dataset) if cls == label])
    # for
    return list_of_labeled_indices
  # get_ways:

  def create_episode(self, n_way, k_shot, n_query):
    support_set, query_set = [], [] # init support_set and query_set
    # add element into them
    for label in n_way:
      support_set.append([self.dataset[idx] for idx in self.ways[label][:k_shot]])
      query_set.append(self.dataset[idx] for idx in self.ways[random.randint(0, len(self.dataset.classes))][:n_query])
    # for
    return FewShotDataset(support_set, self.transform, mode="support"), FewShotDataset(query_set, self.transform, mode="query")
  # make_episode:

  def __getitem__(self, index): return self.episodes[index]
# Episoder():

class FewShotDataset(Dataset):
  def __init__(self, dataset, transform=None, mode="support"):
    self.dataset = dataset
    self.transform = transform
    self.mode = mode
  # __init__():

  def __len__(self): return len(self.dataset)

  def __getitem__(self, index):
    feature, label = self.dataset[index]
    if self.mode == "query": label = -1
    return feature, label
  # __getitem__():
# FewShotDataset():