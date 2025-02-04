import torch

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