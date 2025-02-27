import torch.cuda
import torchvision as tv
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet

def train(DATASET:str, SAVE_TO:str, N_WAY:int, K_SHOT:int, N_QUERY:int, IETRS:int, EPOCHS:int):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform

  # init episode generator
  imageset = tv.datasets.ImageFolder(root=DATASET)
  chosen_classes = list(imageset.class_to_idx.values())[:N_WAY]
  episoder = FewShotEpisoder(imageset, chosen_classes, K_SHOT, N_QUERY, transform)

  # init model
  model_config = {"in_channels": 3, "hidden_channels": 26, "output_channels": 3}
  model = ProtoNet(*model_config.values()).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
  criterion = nn.CrossEntropyLoss()

  progress_bar, whole_loss = tqdm(range(EPOCHS)), float()
  for _ in progress_bar:
    support_set, query_set = episoder.get_episode()
    # STAGE1: compute prototype from support examples
    prototypes = list()
    embedded_features_list = [[] for _ in range(len(support_set.classes))]
    for embedded_feature, label in support_set: embedded_features_list[label].append(embedded_feature)
    for embedded_features in embedded_features_list:
      class_prototype = torch.stack(embedded_features).mean(dim=0)
      prototypes.append(class_prototype.flatten())
    # for
    prototypes = torch.stack(prototypes)
    model.prototyping(prototypes)
    # STAGE2: update parameters form loss associated with prototypes
    epochs_loss = 0.0
    for _ in range(IETRS):
      iter_loss = 0.0
      for feature, label in DataLoader(query_set, shuffle=True):
        loss = criterion(model.forward(feature), label)
        iter_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
      epochs_loss += iter_loss / len(query_set)
    # for # for
    epochs_loss = epochs_loss / IETRS
    progress_bar.set_postfix(loss=epochs_loss)
  # for

  # saving the model's parameters and the other data
  features = {
    "state": model.state_dict(),
    "model_config": model_config,
    "transform": transform,
    "chosen_classes": chosen_classes,
    "framework": (N_WAY, K_SHOT, N_QUERY)
  }  # features
  torch.save(features, SAVE_TO)
  print(f"model save to {SAVE_TO}")
# main()

if __name__ == "__main__": train("../data/omniglot-py/images_background/Futurama", "./model/model.pth", 5, 5, 2, 5, 5)