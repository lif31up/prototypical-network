import random
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

def dist(a: torch.tensor, b: torch.tensor): # euclidean distance
  if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor): raise ValueError(
    "a and b must be pytorch tensor.")
  n, m = a.shape[0], b.shape[0]
  a, b = a.unsqueeze(1).expand(n, m, -1), b.unsqueeze(0).expand(n, m, -1)
  return torch.pow(a - b, 2).sum(2)
# euclidean_distance()

def dist_softmax(target: torch.tensor, q_points: list):
  if not isinstance(target, torch.Tensor): raise ValueError("target must be a pytorch tensor")
  denominator = 0.
  for q_point in q_points: denominator += torch.exp(dist(target, q_point))
  return [torch.exp(dist(target, q_point)) / denominator for q_point in q_points]
# dist_encoder():

def argmax_encoder(result):
  index_of_greatest = 0
  for index, element in enumerate(result):
    if element > index_of_greatest: index_of_greatest = index
  # for
  return torch.tensor([1. if index == index_of_greatest else 0. for index, _ in enumerate(result)])
# encoder():

class FewShotDataset(Dataset):
  """ A custom Dataset class for Few-Shot Learning tasks.
    This dataset can operate in two modes: "support" (for prototype calculation) and "query" (for evaluation). """
  def __init__(self, dataset, n_classes, transform, mode="support"):
    """ Args:
        dataset (list): List of (feature, label) pairs.
        n_classes (int): Number of classes in the dataset.
        transform (callable): Transform to be applied to the features.
        mode (str): Mode of operation, either "support" or "query". Default is "support". """
    self.dataset = [] if not dataset else dataset # Initialize dataset (empty if not provided)
    self.mode, self.transform, self.n_classes = mode, transform, n_classes
    if self.mode == "support": self.prototypes = self.get_prototypes() # Calculate prototypes if in "support" mode
  # __init__():

  def __getitem__(self, idx):
    feature, label = self.dataset[idx]
    return self.transform(feature), label
  # __getitem__():

  def __len__(self): return len(self.dataset)

  def get_prototypes(self):
    """ Calculate prototypes for each class.
        Prototypes are computed as the mean of transformed features for each class.
        * Returns: list: List of prototypes (one per class) if in "support" mode, otherwise an empty list. """
    if not self.mode == "support": return []

    # Group features by class
    ways = [[] for _ in range(self.n_classes)]
    for item in self.dataset:
      feature, label = item
      ways[label].append(item)
    # for

    # Calculate the mean (prototype) for each class
    prototypes = []
    for way in ways:
      prototype = 0.
      for feature, _ in way: prototype += self.transform(feature) # Apply transform and accumulate features
      prototype /= len(way) # Normalize by the number of samples in the class
      prototypes.append(prototype)
    # for
    return prototypes
# FSLDataset()

class FewShotEpisoder:
  """ A class to generate episodes for Few-Shot Learning.
  Each episode consists of a support set and a query set. """
  def __init__(self, dataset, n_way, k_shot, n_query, transform):
    """ Args:
        dataset (Dataset): The base dataset to generate episodes from.
        n_way (int): Number of classes per episode.
        k_shot (int): Number of support samples per class.
        n_query (int): Number of query samples per class.
        transform (callable): Transform to be applied to the features. """
    self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query  # define n-way/k-hot framework parameters
    self.dataset, self.transform = dataset, transform  # init dataset and apply transformer
    self.n_classes, self.ways = self.init()
  # __init__()

  def init(self):
    """ Initialize the class indices for the dataset.
        * Returns: tuple of Number of classes and a list of indices grouped by class.
        * Raises: ValueError: If the dataset is empty or invalid. """
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
    """ Generate an episode consisting of a support set and a query set.
        Returns: tuple of A FewShotDataset for the support set and a FewShotDataset for the query set. """
    support_set, query_set = [], []
    # build support set
    for way in range(0, self.n_classes):
      for cnt, index in enumerate(random.sample(self.ways[way], self.k_shot)): # Randomly sample support indices
        support_set.append(self.dataset[index])
        self.ways[way].pop(cnt) # Add sampled items to the support set
    # build query set
    query_set = support_set.copy()
    for way in range(0, self.n_classes):
      for cnt, index in enumerate(random.sample(self.ways[way], self.n_query)): # Randomly sample query indices
        query_set.append(self.dataset[index])  # Add sampled items to the query set
        self.ways[way].pop(cnt)
    # for for
    # Create FewShotDataset instances for the support and query sets
    return FewShotDataset(support_set, self.n_classes, mode="support", transform=self.transform), FewShotDataset(query_set, self.n_classes, mode="query", transform=self.transform)
  # get_episode()
# Episoder()

# mian function to simulate the few-shot learning pipeline.
def main(path: str):
  # define the image transformations
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform

  # load the dataset
  imageset = tv.datasets.ImageFolder(root=path)
  episoder = FewShotEpisoder(imageset, 3, 4, 4, transform)

  # simulate training flow
  epochs, iters = 2, 2
  for _ in range(epochs):
    support_set, query_set = episoder.get_episode()
    for _ in range(iters):
      for feature, label in DataLoader(support_set, shuffle=True):
        # iterate through the support set
        print(feature.size(), support_set.prototypes[label])
      for feature, label in DataLoader(query_set, shuffle=True):
        # iterate through the query set
        print(feature.size(), label)
  # for for
# main():

if __name__ == "__main__": main("../data/raw/omniglot-py/images_background/Korean")
