import random
import typing
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FewShotDataset(Dataset):
  def __init__(self, dataset, indices: list, classes: list, transform:typing.Callable, mode="support"):
    assert mode in ["support", "query"], "Invalid mode. Must be either 'support' or 'query'." # check if mode is valid
    assert dataset and indices and classes is not None, "Dataset or indices cannot be None." # check if dataset is not None
    self.dataset, self.indices, self.classes = dataset, indices, classes
    self.mode, self.transform = mode, transform
  # __init__

  def __getitem__(self, index: int):
    if index >= len(self.indices):
      raise IndexError("Index out of bounds") # check if index is out of bounds
    feature, label = self.dataset[self.indices[index]]
    feature = self.transform(feature)
    if self.mode == "query": # if mode is query, convert label to one-hot vector
      label = F.one_hot(torch.tensor(self.classes.index(label)), num_classes=len(self.classes)).float()
    return feature, label
  # __getitem__

  def __len__(self): return len(self.indices)
# FSLDataset

class FewShotEpisoder:
  def __init__(self, dataset, classes: list, k_shot: int, n_query: int, transform: typing.Callable):
    assert k_shot > 0 and n_query > 0, "k_shot and n_query must be greater than 0."  # check if k_shot and n_query are valid
    self.k_shot, self.n_query, self.classes = k_shot, n_query, classes
    self.dataset, self.transform = dataset, transform
    self.indices_c = self.get_class_indices()
  # __init__

  def get_class_indices(self):
    indices_c = {label: [] for label in range(self.classes.__len__())}
    for index, (_, label) in enumerate(self.dataset):
      if label in self.classes: indices_c[self.classes.index(label)].append(index)
    for label, _indices_c in indices_c.items():
      indices_c[label] = random.sample(_indices_c, self.k_shot + self.n_query)
    return indices_c
  # get_indices

  def get_episode(self):
    support_examples, query_examples = [], []
    for class_label in range(self.classes.__len__()):
      if len(self.indices_c[class_label]) < self.k_shot + self.n_query: continue  # skip class if it doesn't have enough samples
      selected_indices = random.sample(self.indices_c[class_label], self.k_shot + self.n_query)
      support_examples.extend(selected_indices[:self.k_shot])
      query_examples.extend(selected_indices)
    support_set = FewShotDataset(self.dataset, support_examples, self.classes, self.transform, "support")
    query_set = FewShotDataset(self.dataset, query_examples, self.classes, self.transform, "query")
    return support_set, query_set
  # get_episode
# Episoder