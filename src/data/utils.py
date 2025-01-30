import torch

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
