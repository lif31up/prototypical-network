import random
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision as tv

from config import TRAINING_CONFIG
from src.model.ProtoNet import ProtoNet
from src.FewShotEpisoder import FewShotEpisoder
from tqdm import tqdm

def evaluate(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select device

  # load model
  data = torch.load(MODEL, map_location=device)
  n_way, k_shot, n_query = data["FRAMEWORK"].values()

  # load model
  model = ProtoNet(*data["MODEL_CONFIG"].values()).to(device)
  model.load_state_dict(data["sate"])
  model.eval()

  # create FSL episode generator
  imageset = tv.datasets.ImageFolder(root=DATASET)
  unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), n_way)]
  episoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, data["TRANSFORM"])

  # compute prototype from support examples
  support_set, query_set = episoder.get_episode()
  prototypes = list()
  embedded_features_list = [[] for _ in range(len(support_set.classes))]
  for embedded_feature, label in support_set: embedded_features_list[unseen_classes.index(label)].append(embedded_feature)
  for embedded_features in embedded_features_list:
    class_prototype = torch.stack(embedded_features).mean(dim=0)
    prototypes.append(class_prototype.flatten())
  prototypes = torch.stack(prototypes).to(device, non_blocking=True)
  model.prototyping(prototypes)

  # eval model
  total_loss, count, n_problem = 0., 0, len(query_set)
  criterion = nn.CrossEntropyLoss()
  for feature, label in DataLoader(query_set, shuffle=True, pin_memory=True, num_workers=4):
    feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
    pred = model.forward(feature)
    loss = criterion(pred, label)
    total_loss += loss.item()
    if torch.argmax(pred) == torch.argmax(label): count += 1
  print(f"seen classes: {data['seen_classes']}\nunseen classes: {unseen_classes}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main()

def evaluates(MODEL: str, DATASET: str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # select device

  # load model
  data = torch.load(MODEL, map_location=device)
  n_way, k_shot, n_query = data["FRAMEWORK"].values()
  model = ProtoNet(*data["MODEL_CONFIG"].values()).to(device)
  model.load_state_dict(data["sate"])
  model.eval()
  imageset = tv.datasets.ImageFolder(root=DATASET)
  count, n_problem, n_eval = 0, 0, 10
  progress_bar = tqdm(range(n_eval))
  for _ in progress_bar:
    # create FSL episode generator
    unseen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), n_way)]
    evisoder = FewShotEpisoder(imageset, unseen_classes, k_shot, n_query, data["TRANSFORM"])

    # compute prototype from support examples
    support_set, query_set = evisoder.get_episode()
    prototypes = list()
    embedded_features_list = [[] for _ in range(len(support_set.classes))]
    for embedded_feature, label in support_set: embedded_features_list[unseen_classes.index(label)].append(
      embedded_feature)
    for embedded_features in embedded_features_list:
      class_prototype = torch.stack(embedded_features).mean(dim=0)
      prototypes.append(class_prototype.flatten())
    prototypes = torch.stack(prototypes).to(device, non_blocking=True)
    model.prototyping(prototypes)

    # eval model
    total_loss = 0.
    criterion = nn.CrossEntropyLoss()
    for feature, label in DataLoader(query_set, shuffle=True, pin_memory=True, num_workers=4):
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      pred = model.forward(feature)
      loss = criterion(pred, label)
      total_loss += loss.item()
      if torch.argmax(pred) == torch.argmax(label): count += 1
      n_problem += 1
    progress_bar.set_postfix(accuracy=count / n_problem)
  print(f"seen classes: {data['seen_classes']}\naccuracy: {count / n_problem:.4f}({count}/{n_problem})")
# main()

if __name__ == "__main__": evaluates("./model/5w5s.bin", "../data/omniglot-py/images_background/Futurama")