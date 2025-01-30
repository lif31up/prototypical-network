from torch.utils.data import Dataset

class FewShotEpisoder:
  def __init__(self, dataset, n_way, k_shot, n_query, transform=None):
    self.dataset = dataset
    self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
    self.transform = transform
    self.episodes = [self.create_episode(n_way, k_shot, n_query) for _ in range(0,4)]
  # __init__()

  def create_episode(self, n_way, k_shot, n_query):
    support_set = FewShotDataset([self.dataset[idx] for idx in range(0, k_shot)], self.transform, mode="support")
    query_set = FewShotDataset([self.dataset[idx] for idx in range(0, n_query)], self.transform, mode="query")
    return support_set, query_set
  # make_episode

  def __getitem__(self, index): return self.episodes[index]
# Episoder()

class FewShotDataset(Dataset):
  def __init__(self, dataset, transform=None, mode="support"):
    self.dataset = dataset
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