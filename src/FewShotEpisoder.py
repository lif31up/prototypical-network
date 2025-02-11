import random
import typing
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class FewShotDataset(Dataset):
  """ A custom Dataset class for Few-Shot Learning tasks.
    This dataset can operate in two modes: "support" (for prototype calculation) and "query" (for evaluation). """
  def __init__(self, dataset: typing.Iterable, indices: list, classes: list, transform:typing.Callable, mode="support"):
    """ Args:
        dataset (list): List of (feature, label) pairs.
        indices (list): List of indices to be used for the dataset.
        transform (callable): Transform to be applied to the features.
        mode (str): Mode of operation, either "support" or "query". Default is "support". """
    assert mode in ["support", "query"], "Invalid mode. Must be either 'support' or 'query'." # check if mode is valid
    assert dataset and indices and classes is not None, "Dataset or indices cannot be None." # check if dataset is not None

    self.dataset, self.indices, self.classes = dataset, indices, classes
    self.mode, self.transform = mode, transform
  # __init__():

  def __getitem__(self, index: int):
    """ Returns a sample from the dataset at the given index.
        Args: index of the sample to be retrieved.
        Returns: tuple of the transformed feature and the label. """
    if index >= len(self.indices):
      raise IndexError("Index out of bounds") # check if index is out of bounds
    feature, label = self.dataset[self.indices[index]]
    # apply transformation
    feature = self.transform(feature)
    if self.mode == "query": # if mode is query, convert label to one-hot vector
      label = F.one_hot(torch.tensor(label), num_classes=len(self.classes)).float()
    return feature, label
  # __getitem__():

  def __len__(self): return len(self.indices)
# FSLDataset()

class FewShotEpisoder:
  """ A class to generate episodes for Few-Shot Learning.
  Each episode consists of a support set and a query set. """
  def __init__(self, dataset, classes: list, k_shot: int, n_query: int, transform: typing.Callable):
    """ Args:
        dataset (Dataset): The base dataset to generate episodes from.
        k_shot (int): Number of support samples per class.
        n_query (int): Number of query samples per class.
        transform (callable): Transform to be applied to the features. """
    assert k_shot > 0 and n_query > 0, "k_shot and n_query must be greater than 0."  # check if k_shot and n_query are valid

    self.k_shot, self.n_query, self.classes = k_shot, n_query, classes
    self.dataset, self.transform = dataset, transform
    self.indices_c = self.get_class_indices()
  # __init__()

  def get_class_indices(self) -> dict:
    """ Initialize the class indices for the dataset.
        Returns: tuple of Number of classes and a list of indices grouped by class. """
    indices_c = {label: [] for label in range(len(self.classes))}
    for index, (_, label) in enumerate(self.dataset):
      if label in self.classes: indices_c[label].append(index)
    return indices_c
  # get_indices():

  def get_episode(self) -> tuple:  # select classes using list of chosen indexes
    """ Generate an episode consisting of a support set and a query set.
        Returns: tuple of A FewShotDataset for the support set and a FewShotDataset for the query set. """
    # get support and query examples
    support_examples, query_examples = [], []
    for class_label in self.classes:
      if len(self.indices_c[class_label]) < self.k_shot + self.n_query: continue # skip class if it doesn't have enough samples
      selected_indices = random.sample(self.indices_c[class_label], self.k_shot + self.n_query)
      support_examples.extend(selected_indices[:self.k_shot])
      query_examples.extend(selected_indices)

    # init support and query datasets
    support_set = FewShotDataset(self.dataset, support_examples, self.classes, self.transform, "support")
    query_set = FewShotDataset(self.dataset, query_examples, self.classes, self.transform, "query")

    return support_set, query_set
  # get_episode()
# Episoder()