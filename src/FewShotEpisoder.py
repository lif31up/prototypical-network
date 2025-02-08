import typing
import torch
import torchvision as tv
from torch.utils.data import Dataset

class FewShotDataset(Dataset):
  """ A custom Dataset class for Few-Shot Learning tasks.
    This dataset can operate in two modes: "support" (for prototype calculation) and "query" (for evaluation). """
  def __init__(self, dataset, indices, transform, mode="support"):
    """ Args:
        dataset (list): List of (feature, label) pairs.
        indices (list): List of indices to be used for the dataset.
        transform (callable): Transform to be applied to the features.
        mode (str): Mode of operation, either "support" or "query". Default is "support". """
    self.dataset, self.indices = [] if not dataset else dataset, indices # Initialize dataset (empty if not provided)
    self.mode, self.transform = mode, transform
    self.classes = dataset.classes
  # __init__():

  def __getitem__(self, index):
    if index >= len(self.indices): raise IndexError("Index out of bounds")
    feature, label = self.dataset[self.indices[index]]
    if self.mode == "query":
      one_hot_vector = torch.zeros(len(self.classes))
      one_hot_vector[label] = 1.
      label = one_hot_vector.requires_grad_(True)
    return self.transform(feature), label
  # __getitem__():

  def __len__(self): return len(self.indices)

# FSLDataset()

class FewShotEpisoder:
  """ A class to generate episodes for Few-Shot Learning.
  Each episode consists of a support set and a query set. """
  def __init__(self, dataset: tv.datasets.ImageFolder, k_shot: int, n_query: int, transform: typing.Callable):
    """ Args:
        dataset (Dataset): The base dataset to generate episodes from.
        k_shot (int): Number of support samples per class.
        n_query (int): Number of query samples per class.
        transform (callable): Transform to be applied to the features. """
    self.k_shot, self.n_query = k_shot, n_query  # define n-way/k-hot framework parameters
    self.dataset, self.transform = dataset, transform  # init dataset and apply transformer
    self.indices_c = self.get_indices()
  # __init__()

  def get_indices(self):
    """ Initialize the class indices for the dataset.
        * Returns: tuple of Number of classes and a list of indices grouped by class. """
    indices_c = [[] for _ in range(len(self.dataset.classes))]
    for index, (feature, label) in enumerate(self.dataset): indices_c[label].append(index)
    return indices_c
  # get_indices():

  def get_episode(self):  # select classes using list of chosen indexes
    """ Generate an episode consisting of a support set and a query set.
        Returns: tuple of A FewShotDataset for the support set and a FewShotDataset for the query set. """
    buffer_indices_c = self.indices_c.copy()
    support_examples, query_examples = [], []
    # select support examples
    for index_indices, indices in enumerate(buffer_indices_c):
      for index, x_index in enumerate(indices):
        support_examples.append(x_index)
        buffer_indices_c[index_indices].pop(index)
    # select query examples
    query_examples = support_examples.copy()
    for index_indices, indices in enumerate(buffer_indices_c):
      for index, x_index in enumerate(indices):
        support_examples.append(x_index)
        buffer_indices_c[index_indices].pop(index)
    # init datasets
    support_set = FewShotDataset(self.dataset, support_examples, mode="support", transform=self.transform)
    query_set = FewShotDataset(self.dataset, query_examples, mode="query", transform=self.transform)

    return support_set, query_set
  # get_episode()
# Episoder()
