import random
import torch
import torchvision as tv
from torch.utils.data import Dataset, DataLoader

def dist(tensor_a: torch.tensor, tensor_b: torch.tensor, metric: str = "euclidean") -> float: # euclidean distance
  if tensor_a.shape != tensor_b.shape: raise ValueError("input tensors must have the same shape.")

  # flatten the tensors to handle n-dimensional inputs
  tensor_a_flat = tensor_a.view(-1)
  tensor_b_flat = tensor_b.view(-1)

  # calculate distance based on the speicfied metric
  if metric == "euclidean":
    distance = torch.norm(tensor_a_flat - tensor_b_flat, p=2).item()
  elif metric == "manhattan":
    distance = torch.norm(tensor_a_flat - tensor_b_flat, p=1).item()
  elif metric == "cosine":
    dot_product = torch.dot(tensor_a_flat, tensor_b_flat)
    norm_tensor_a = torch.norm(tensor_a_flat, p=2)
    norm_tensor_b = torch.norm(tensor_b_flat, p=2)
    cosine_sim = dot_product / (norm_tensor_a * norm_tensor_b)
    distance = 1 - cosine_sim.item()  # Convert similarity to distance
  else:
    raise ValueError(f"uspported metric: {metric}. Choose from 'euclidean', 'manhattan', or 'cosine'.")
  # if elif else
  return distance
# dist()

def softmax_dist(tensor: torch.tensor, q_points: torch.tensor, temperature=1.0) -> torch.Tensor:
  distances = torch.norm(q_points - tensor, dim=1, p=2)
  exp_values = torch.exp(-distances / temperature)
  softmax_distances = exp_values / torch.sum(exp_values)
  return softmax_distances
# softmax_dist():

def one_hot_encoder(softmax_probs: torch.Tensor):
  one_hot = torch.zeros_like(softmax_probs)
  one_hot[torch.argmax(softmax_probs)] = 1.0
  return one_hot
# one_hot_encoder():

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
