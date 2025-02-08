import random
import typing

import torchvision.datasets
from tqdm import tqdm
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

class FewShotDataset(Dataset):
  """ A custom Dataset class for Few-Shot Learning tasks.
    This dataset can operate in two modes: "support" (for prototype calculation) and "query" (for evaluation). """
  def __init__(self, dataset, indices, transform, mode="support"):
    """ Args:
        dataset (list): List of (feature, label) pairs.
        n_classes (int): Number of classes in the dataset.
        transform (callable): Transform to be applied to the features.
        mode (str): Mode of operation, either "support" or "query". Default is "support". """
    self.dataset, self.indices = [] if not dataset else dataset, indices # Initialize dataset (empty if not provided)
    self.mode, self.transform = mode, transform
    self.classes = dataset.classes
  # __init__():

  def __getitem__(self, index):
    if index >= len(self.indices): raise IndexError("Index out of bounds")
    feature, label = self.dataset[self.indices[index]]
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
        n_way (int): Number of classes per episode.
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
        prototypes.append(sum)
      # update loss
      for embedded_feature, label in DataLoader(query_set, shuffle=True):
        print(embedded_feature.size(), prototypes[label].size())
  # for for
# main():

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean")
