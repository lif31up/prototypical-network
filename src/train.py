import random
import torch.cuda
import torchvision as tv
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.FewShotEpisoder import FewShotEpisoder
from src.model.ProtoNet import ProtoNet
from config import TRAINING_CONFIG, HYPER_PARAMETERS, MODEL_CONFIG, FRAMEWORK
from safetensors.torch import save_file

def train(DATASET:str, SAVE_TO:str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # init device

  # define transform
  transform = tv.transforms.Compose([
    tv.transforms.Resize((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]) # transform

  # init episode generator
  imageset = tv.datasets.ImageFolder(root=DATASET)
  seen_classes = [_ for _ in random.sample(list(imageset.class_to_idx.values()), FRAMEWORK['n_way'])]
  episoder = FewShotEpisoder(imageset, seen_classes, FRAMEWORK['k_shot'], FRAMEWORK['n_query'], transform)

  # init model
  model = ProtoNet(*MODEL_CONFIG.values()).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=HYPER_PARAMETERS["lr"], weight_decay=HYPER_PARAMETERS["weight_decay"])
  criterion = nn.CrossEntropyLoss()

  progress_bar, whole_loss = tqdm(range(TRAINING_CONFIG['epochs'])), float()
  for _ in progress_bar:
    support_set, query_set = episoder.get_episode()
    # STAGE1: compute prototype from support examples
    prototypes = list()
    embedded_features_list = [[] for _ in range(len(support_set.classes))]
    for embedded_feature, label in support_set: embedded_features_list[seen_classes.index(label)].append(embedded_feature)
    for embedded_features in embedded_features_list:
      class_prototype = torch.stack(embedded_features).mean(dim=0)
      prototypes.append(class_prototype.flatten())
    # for
    prototypes = torch.stack(prototypes)
    model.prototyping(prototypes)
    # STAGE2: update parameters form loss associated with prototypes
    epochs_loss = 0.0
    for _ in range(TRAINING_CONFIG['iters']):
      iter_loss = 0.0
      for feature, label in DataLoader(query_set, batch_size=TRAINING_CONFIG["batch_size"], shuffle=True, pin_memory=True, num_workers=4):
        loss = criterion(model.forward(feature), label)
        iter_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
      epochs_loss += iter_loss / len(query_set)
    # for # for
    epochs_loss = epochs_loss / TRAINING_CONFIG['iters']
    progress_bar.set_postfix(loss=epochs_loss)
  # for

  # saving model
  features = {
    "sate": model.state_dict(),
    "FRAMEWORK": FRAMEWORK,
    "MODEL_CONFIG": MODEL_CONFIG,
    "HYPER_PARAMETERS": HYPER_PARAMETERS,
    "TRAINING_CONFIG": TRAINING_CONFIG,
    "TRANSFORM": transform,
    "seen_classes": seen_classes
  }  # feature
  torch.save(features, SAVE_TO)
  save_file(model.state_dict(), SAVE_TO.replace(".pth", ".safetensors"))
# main()

if __name__ == "__main__": train("../data/omniglot-py/images_background/Futurama", "./model/model.pth")