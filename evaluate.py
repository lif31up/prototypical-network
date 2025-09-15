import torch
from torch.utils.data import DataLoader
import random
from FewShotEpisoder import FewShotEpisoder
from config import Config
from model.ProtoNet import get_prototypes, ProtoNet

def evaluate(model, evisoder, device, logging=True):
  model.eval()
  support_set, query_set = evisoder.get_episode()
  model.prototypes = get_prototypes(support_set).to(device)
  n_counts, n_problems = 0, len(query_set)
  for feature, label in DataLoader(query_set, shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    pred = model.forward(feature)
    if torch.argmax(pred) == torch.argmax(label): n_counts += 1
  if logging: print(f"unseen classes: {evisoder.classes}\naccuracy: {n_counts / n_problems:.4f}({n_counts}/{n_problems})")
  return n_counts, n_problems
# evluate

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  protonet_config = Config()
  my_data = torch.load(
    f='/content/drive/MyDrive/Colab Notebooks/PRN.bin',
    weights_only=False,
    map_location=torch.device('cpu'))
  my_model = ProtoNet(my_data["config"]).to(device)
  my_model.load_state_dict(my_data["state"])
  imageset = protonet_config.imageset
  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), protonet_config.n_way)]
  evisoder = FewShotEpisoder(imageset, unseen_classes, protonet_config.k_shot, protonet_config.n_query,
                             protonet_config.transform)
  evaluate(my_model, my_data, device)