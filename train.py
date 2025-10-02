import random
import torch.cuda
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import Config
from FewShotEpisoder import FewShotEpisoder
from model.ProtoNet import ProtoNet, get_prototypes


def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
  try:
    if m.bias is not None: nn.init.zeros_(m.bias)
  except: pass
# init_weights

def train(model, path, config:Config, episoder:FewShotEpisoder, device, init=True):
  model.to(device)
  if init: model.apply(init_weights)
  optim = torch.optim.SGD(model.parameters(), lr=config.alpha)
  criterion = nn.CrossEntropyLoss()

  progression = tqdm(range(config.epochs))
  for _ in progression:
    support_set, query_set = episoder.get_episode()
    prototypes = get_prototypes(support_set)
    for _ in range(config.iterations):
      iter_loss = float(0)
      for feature, label in DataLoader(query_set, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4):
        feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
        pred = model.forward(feature, prototypes)
        loss = criterion(pred, label)
        optim.zero_grad()
        loss.backward()
        if config.clip_grad: nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        iter_loss += loss.item()
      iter_loss /= len(query_set)
      progression.set_postfix(iter_loss=iter_loss)
    del prototypes, feature, label, pred, loss
    torch.cuda.empty_cache()

  features = {
    "state": model.state_dict(),
    "config": config,
  } # features
  torch.save(features, f'{path}.bin')
# main

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  protonet_config = Config()
  imageset = protonet_config.imageset
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), protonet_config.n_way)]
  episoder = FewShotEpisoder(imageset, seen_classes, protonet_config.k_shot, protonet_config.n_query, protonet_config.transform)
  model = ProtoNet(protonet_config)
  train(model=model, path=protonet_config.save_to, config=protonet_config, episoder=episoder, device=device, init=True)
# if __name__ = "__main__"